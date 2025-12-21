import math
from pathlib import Path
from typing import Optional
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


class TripletDataset(Dataset):
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

        # For a minor optimization
        self._total_indexable_len: int = sum(self._indexable_lens.values())

    def __len__(self) -> int:
        return self._total_indexable_len

    def __getitem__(self, index):
        current_segment, actual_index = self._index_dataset(index)
        entry = current_segment[actual_index]

        positive = entry["positive"]
        anchor = random.choice(entry["group"])

        negative_index = index
        while index <= negative_index < index + self._context_len:
            negative_index = random.randint(0, self.__len__() - 1)
        ns, ni = self._index_dataset(negative_index)
        negative = ns[ni]["positive"]
        return anchor, positive, negative

    def _index_dataset(self, index) -> tuple[datasets.Dataset, int]:
        actual_index: int = index
        segment_name: Optional[str] = None
        for k, l in self._indexable_lens.items():
            if actual_index < l:
                segment_name = k
                break
            actual_index -= l
        else:
            raise IndexError("Index out of range.")

        segment = self._hf_dataset[segment_name]
        return segment, actual_index
