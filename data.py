import math
from pathlib import Path
from typing import Optional
import warnings
from datasets import DatasetDict, load_dataset
import datasets
import random
import torch
from torch.utils.data import Dataset
from datetime import datetime
from ast import literal_eval


def fix_surrogates(s: str) -> str:
    return s.encode("utf-16", "surrogatepass").decode("utf-16")


def eval_group(entry):
    entry["group"] = literal_eval(entry["group"])
    entry["group"] = [fix_surrogates(x) for x in entry["group"]]
    return entry


def load_and_split(
    files: list[Path],
    train_size: float = 0.85,
    timestamp: Optional[datetime] = None,
) -> DatasetDict:
    assert 0.0 <= train_size <= 1, "Train size must be between 0 and 1"
    segments = {
        fname.stem: fname.__str__()
        for fname in files
    }
    dataset = load_dataset('parquet', data_files=segments)
    assert isinstance(dataset, DatasetDict)
    train_dataset: DatasetDict = DatasetDict()
    val_dataset: DatasetDict = DatasetDict()
    for seg_name, segment in dataset.items():
        if timestamp:
            segment = segment.filter(
                lambda x: datetime.fromisoformat(x["timestamp"]) > timestamp
            )
        n = segment.__len__()
        split_idx: int = int(n * train_size)
        train_dataset[seg_name] = segment.select(range(0, split_idx)).add_column("segment_id", [seg_name]*split_idx)
        val_dataset[seg_name] = segment.select(range(split_idx, n)).add_column("segment_id", [seg_name]*(n - split_idx))

    return DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
    })


def collate_triplet(batch: list[tuple[str, str, str]]) -> tuple[list[str], list[str], list[str]]:
    anchors, positives, negatives = zip(*batch)
    return anchors, positives, negatives


def collate_infonce(batch: list[tuple[str, list[str]]]) -> tuple[list[str], list[list[str]]]:
    anchors, positives = zip(*batch)
    return anchors, positives


class TripletDataset(Dataset):
    def __init__(
        self,
        hf_dataset: datasets.DatasetDict,
        context_len: int,
        full_context: bool = False,
        last_message_only: bool = False,
        any_message_prob: float = 0.1,
    ) -> None:
        """
        Create a TripletDataset object

        A TripletDataset object returns (anchor, positive, negative) triplets when indexed.

        Parameters
        ----------
        hf_dataset
            Underlying huggingface dataset
        context_len
            Context Length
        full_context
            Set True to sample all sub-messages individually as anchors. Otherwise anchor is randomly sampled. (Produces context_len times more elements.)
        last_message_only
            Only return the last message of block as its anchor.
        any_message_prob
            Probability of randomly selecting a message even though last_message_only is provided. Used for regularization.
        """  # noqa
        super().__init__()

        self._segments: list[str] = list(hf_dataset.keys())
        self._hf_dataset = hf_dataset
        self._context_len: int = context_len
        self._full_context: bool = full_context
        self._last_message_only: bool = last_message_only
        self._any_message_prob: float = any_message_prob
        if self._full_context and self._last_message_only:
            warnings.warn("full_context and only_last_message are both True. Ignoring only_last_message.")

        self._indexable_lens: dict[str, int] = {}
        for k in self._segments:
            v = self._hf_dataset[k]
            self._indexable_lens[k] = v.num_rows

        if self._full_context:
            self._total_indexable_len: int = sum(self._indexable_lens.values()) * self._context_len
        else:
            self._total_indexable_len: int = sum(self._indexable_lens.values())

        # Maps block indexes to (segment name, actual block index) pairs
        self.__segment_cache: dict[int, tuple[str, int]] = {}

    def __len__(self) -> int:
        return self._total_indexable_len

    def __getitem__(self, index: int):
        if self._full_context:
            block_idx: int = index // self._context_len
            group_idx: int = index % self._context_len
        else:
            block_idx = index
            # TODO: Maybe make any message a special case when any_message_prob=1
            last_message_only = self._last_message_only and (random.uniform(0.0, 1.0) >= self._any_message_prob)
            if last_message_only:
                group_idx = self._context_len - 1
            else:
                group_idx = random.randint(0, self._context_len - 1)
        current_segment, actual_block_index = self._index_dataset(block_idx)
        entry = current_segment[actual_block_index]

        positive = entry["positive"]
        anchor = entry["group"][group_idx]

        negative_index = block_idx
        while block_idx - self._context_len <= negative_index < block_idx + self._context_len:
            negative_index = random.randint(0, self.__len__() // self._context_len - 1)
        ns, ni = self._index_dataset(negative_index)
        negative = ns[ni]["positive"]
        return anchor, positive, negative

    def _index_dataset(self, block_index) -> tuple[datasets.Dataset, int]:
        pair: Optional[tuple[str, int]] = self.__segment_cache.get(block_index, None)
        if pair is None:
            actual_block_index: int = block_index
            segment_name: Optional[str] = None
            for k, l in self._indexable_lens.items():
                if actual_block_index < l:
                    segment_name = k
                    break
                actual_block_index -= l
            else:
                raise IndexError("Index out of range.")

            self.__segment_cache[block_index] = (segment_name, actual_block_index)
        else:
            segment_name, actual_block_index = pair

        segment = self._hf_dataset[segment_name]
        return segment, actual_block_index


class MultipositiveInfoNCEDataset(Dataset):
    def __init__(
        self,
        hf_dataset: datasets.DatasetDict,
        context_len: int,
    ) -> None:
        super().__init__()

        self._segments: list[str] = list(hf_dataset.keys())
        self._hf_dataset = hf_dataset
        self._context_len: int = context_len

        self._indexable_lens: dict[str, int] = {}
        for k in self._segments:
            v = self._hf_dataset[k]
            self._indexable_lens[k] = v.num_rows - self._context_len + 1

        self._total_indexable_len: int = sum(self._indexable_lens.values())
        self.__segment_cache: dict[int, tuple[str, int]] = {}

    def __len__(self) -> int:
        return self._total_indexable_len

    def __getitem__(self, index: int):
        block_idx = index

        current_segment, actual_block_index = self._index_dataset(block_idx)
        entries = current_segment[actual_block_index:actual_block_index + self._context_len]

        # Anchor message is the last message of the first block
        # Positives are all blocks containing the anchor message
        anchor = entries["group"][0][-1]
        positives = entries["positive"]
        return anchor, positives

    def _index_dataset(self, block_index) -> tuple[datasets.Dataset, int]:
        pair: Optional[tuple[str, int]] = self.__segment_cache.get(block_index, None)
        if pair is None:
            actual_block_index: int = block_index
            segment_name: Optional[str] = None
            for k, l in self._indexable_lens.items():
                if actual_block_index < l:
                    segment_name = k
                    break
                actual_block_index -= l
            else:
                raise IndexError("Index out of range.")

            self.__segment_cache[block_index] = (segment_name, actual_block_index)
        else:
            segment_name, actual_block_index = pair

        segment = self._hf_dataset[segment_name]
        return segment, actual_block_index
