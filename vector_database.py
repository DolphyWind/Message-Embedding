from typing import Iterable
import numpy as np
from pathlib import Path
import pickle


class VectorDatabase:
    def __init__(self, dim: int) -> None:
        self.dim: int = dim
        self._vectors: dict[int, np.ndarray] = {}

    def add(self, vid: int, vector: Iterable[float]) -> None:
        if vid in self._vectors:
            raise ValueError(f"Duplicate id: {vid}")

        v = np.asarray(vector, dtype=np.float64)
        if v.shape != (self.dim,):
            raise ValueError(f"Expected vector of shape ({self.dim},)")

        self._vectors[vid] = v

    def remove(self, vid: int) -> None:
        del self._vectors[vid]

    def get(self, vid: int) -> np.ndarray:
        return self._vectors[vid]

    def search(
        self,
        query: Iterable[float],
        k: int,
    ) -> list[tuple[int, float]]:
        if k <= 0:
            return []

        q = np.asarray(query, dtype=np.float64)
        if q.shape != (self.dim,):
            raise ValueError(f"Expected query of shape ({self.dim},)")

        ids = list(self._vectors.keys())
        if not ids:
            return []

        mat = np.stack([self._vectors[i] for i in ids])
        mat_norms = np.linalg.norm(mat, axis=1)
        q_norm = np.linalg.norm(q)

        if q_norm == 0.0:
            raise ValueError("Query vector has zero norm")

        sims = (mat @ q) / (mat_norms * q_norm)
        idx = np.argpartition(-sims, min(k, len(sims)) - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]

        return [(ids[i], float(sims[i])) for i in idx]

    def __len__(self) -> int:
        return len(self._vectors)

    def save(self, path: str | Path):
        with open(path, 'wb') as f:
            pickle.dump(self._vectors, f)

    def load(self, path: str | Path):
        with open(path, 'rb') as f:
            self._vectors = pickle.load(f)
