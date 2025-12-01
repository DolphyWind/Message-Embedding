import math
from pathlib import Path
from typing import Optional
from datasets import DatasetDict, load_dataset
import datasets
import random
import torch
from torch.utils.data import Dataset


def load_and_split(files: list[Path], train_size: float = 0.85) -> DatasetDict:
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
        n = segment.__len__()
        split_idx: int = int(n * train_size)
        train_dataset[seg_name] = segment.select(range(0, split_idx)).add_column("segment_id", [seg_name]*split_idx)
        val_dataset[seg_name] = segment.select(range(split_idx, n)).add_column("segment_id", [seg_name]*(n - split_idx))

    return DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
    })

def collate_triplet(batch: list[tuple[str, str, str, float]]) -> tuple[list[str], list[str], list[str], torch.Tensor]:
    anchors = [x[0] for x in batch]
    positives = [x[1] for x in batch]
    negatives = [x[2] for x in batch]
    margins = torch.tensor([x[3] for x in batch])
    return anchors, positives, negatives, margins


class TripletDataset(Dataset):
    def __init__(
        self,
        hf_dataset: datasets.DatasetDict,
        context_len: int,
        in_context_probability: float = 0.05,
        base_margin: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self._segments: list[str] = list(hf_dataset.keys())
        self._hf_dataset = hf_dataset
        self._context_len: int = context_len
        self._icp: float = in_context_probability
        self._base_margin: float = base_margin
        self._alpha: float = alpha

        self._indexable_lens: dict[str, int] = {}
        for k in self._segments:
            v = self._hf_dataset[k]
            self._indexable_lens[k] = v.num_rows - self._context_len + 1

        # For a minor optimization
        self._total_indexable_len: int = sum(self._indexable_lens.values())

    def __len__(self) -> int:
        return 1000
        return self._total_indexable_len

    def __getitem__(self, index):
        current_segment, actual_index = self._index_dataset(index)
        context = current_segment[actual_index:actual_index + self._context_len]

        in_context: bool = random.uniform(0.0, 1.0) < self._icp
        negative_index: int
        margin: float
        if in_context:
            negative_index = random.randint(0, self._context_len - 2)
            negative_example = current_segment[actual_index + negative_index]
            C_minus_1 = self._context_len - 1
            margin = self._base_margin * (1 - math.exp(-self._alpha * (C_minus_1 - negative_index)))
        else:
            negative_index = index
            while index <= negative_index < index + self._context_len:
                negative_index = random.randint(0, self.__len__() - 1)
            ns, ni = self._index_dataset(negative_index)
            negative_example = ns[ni]
            margin = self._base_margin

        positive_example = current_segment[actual_index + self._context_len - 1]
        anchor = context["content"]
        anchor_sentence = ''.join([
            f"<user{i}>{s}</user>"
            for i, s in enumerate(anchor)
        ])
        return anchor_sentence, positive_example["content"], negative_example["content"], margin

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
