from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

from datasets import DatasetDict, Dataset
from faiss import IndexFlatIP, IndexIDMap2, read_index
import numpy as np
import torch
from tqdm import tqdm

from data import load_and_split, eval_group
from model import MessageEmbeddingModel


def parse_args():
    parser = ArgumentParser(
        description="Test a model.",
    )
    parser.add_argument(
        '--emb_path',
        type=Path,
        help="",
        required=True,
    )
    return parser.parse_args()


class Tester:
    def __init__(self, args) -> None:
        self.emb_path: Path = args.emb_path
        metadata: dict[str, Any] = {}
        with open(self.emb_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        self.timestamp: Optional[str] = metadata["timestamp"]
        self.data_path: Path = Path(metadata["data_path"])
        self.model_path: Path = Path(metadata["model_path"])
        self.train_split: float = metadata["train_split"]

        train_state = torch.load(self.model_path / 'train_state.pth', weights_only=False)
        train_args = train_state['args']
        self.model = MessageEmbeddingModel(
            base_model=train_args.base_model,
            message_context_length=train_args.context_length,
            pooling_mode=train_args.pooling_mode,
            use_lora=train_args.lora,
            lora_config={
                "r": train_args.lora_rank,
                "lora_alpha": train_args.lora_alpha,
                "target_modules": ["query", "key", "value", "output.dense"],
                "bias": "none",
                "lora_dropout": train_args.lora_dropout,
            },
            initialize_new=True,
        )
        model_state_dict = torch.load(self.model_path / 'model_best.pth')['model']
        self.model.load_state_dict(model_state_dict)
        del model_state_dict

        files = [p for p in self.data_path.glob('*.parquet')]
        self.data: DatasetDict = load_and_split(
            files,
            self.train_split,
            timestamp=datetime.fromisoformat(self.timestamp) if self.timestamp else None
        )["val"]
        for k in self.data:
            self.data[k] = self.data[k].map(eval_group, num_proc=train_args.num_workers)

        self.vector_db = IndexFlatIP(self.model.embedding_dim)
        self.vector_db = IndexIDMap2(self.vector_db)
        self.vector_db = read_index(str(self.emb_path / "embeddings.faiss"))
        self.device = 'cuda'

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        batch_size: int = 4

        results: dict[str, dict] = defaultdict(dict)
        total_ta1: int = 0
        total_ta5: int = 0
        total_ta8: int = 0
        total_len: int = 0
        for dataset_name, v in self.data.items():
            top_at_1: int = 0
            top_at_5: int = 0
            top_at_8: int = 0
            total_sentence_length = 0
            loop = tqdm(range(0, len(v), batch_size))
            loop.set_description(dataset_name)

            for i in loop:
                last_idx = min(i + batch_size, len(v))
                batch = v[i:last_idx]
                true_indices: list[int] = []
                sentences: list[str] = []
                # sentences: list[str] = [g[-1] for g in batch['group']]
                for g, idx in zip(batch['group'], batch['index']):
                    for sent in g:
                        if len(sent) > 7 and ' ' in sent:
                            sentences.append(sent)
                            true_indices.append(idx)
                inputs = self.model.tokenizer(
                    sentences,
                    padding=True,
                    truncation=True,
                    max_length=self.model.token_context_length,
                    return_tensors='pt',
                )
                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                }
                embedding = self.model(**inputs)
                np_arr = embedding.detach().cpu().numpy()
                norms = np.linalg.norm(np_arr, axis=1, keepdims=True)
                np_arr = np_arr / norms
                # true_indices = batch['index']

                pred_indices = self.vector_db.search(np_arr, k=8)[1]
                top_1: int = 0
                top_5: int = 0
                top_8: int = 0
                for j in range(len(true_indices)):
                    top_1 += int(true_indices[j] in pred_indices[j, :1])
                    top_5 += int(true_indices[j] in pred_indices[j, :5])
                    top_8 += int(true_indices[j] in pred_indices[j, :8])
                top_at_1 += top_1
                top_at_5 += top_5
                top_at_8 += top_8
                total_sentence_length += len(sentences)

            total_ta1 += top_at_1
            total_ta5 += top_at_5
            total_ta8 += top_at_8
            current_results = {
                "top_at_1": top_at_1 / total_sentence_length,
                "top_at_5": top_at_5 / total_sentence_length,
                "top_at_8": top_at_8 / total_sentence_length,
            }
            total_len += total_sentence_length
            print(json.dumps(current_results, indent=4))
            results["datasets"][str(dataset_name)] = current_results
        results["total"] = {
            "top_at_1": total_ta1 / total_len,
            "top_at_5": total_ta5 / total_len,
            "top_at_8": total_ta8 / total_len,
        }

        with open(self.emb_path / "results.json", "w") as f:
            json.dump(results, f, indent=4)
        print(json.dumps(results, indent=4))


def main():
    args = parse_args()
    tester = Tester(args)
    tester.test()


if __name__ == "__main__":
    main()
