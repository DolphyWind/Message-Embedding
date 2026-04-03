# Wang et. al. 2022
# https://arxiv.org/abs/2005.10242

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import random

from datasets import DatasetDict
import numpy as np
import torch
from tqdm import tqdm

from data import load_and_split
from model import MessageEmbeddingModel


def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean()  # .log()


def parse_args():
    parser = ArgumentParser(
        description="Calculate alignment and uniformity losses.",
    )
    parser.add_argument(
        '--data_path',
        type=Path,
        default=Path('./data/'),
        help="Input data for the model.",
    )
    parser.add_argument(
        '--model_path',
        type=Path,
        help="Model path",
        required=True,
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        help="Timestamp to filter",
        default=None
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=True,
    )
    return parser.parse_args()


class LossCalculator:
    def __init__(self, args) -> None:
        self.data_path: Path = args.data_path
        self.model_path: Path = args.model_path
        self.timestamp: str = args.timestamp
        self.batch_size: int = args.batch_size
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
            }
        )
        model_state_dict = torch.load(self.model_path / 'model_best.pth')['model']
        self.model.load_state_dict(model_state_dict)
        del model_state_dict

        files = [p for p in self.data_path.glob('*.parquet')]
        self.data: DatasetDict = load_and_split(
            files,
            0.0,
            timestamp=datetime.fromisoformat(self.timestamp) if self.timestamp else None
        )["val"]

        self.device = 'cuda'

    def calculate_losses(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            uniformity_losses: list[float] = []
            for k, v in self.data.items():
                data_indices = list(range(0, len(v)))
                random.shuffle(data_indices)
                loop = tqdm(range(0, len(v), self.batch_size))
                loop.set_description(k)
                for i in loop:
                    last_idx = min(i + self.batch_size, len(v))
                    indices = data_indices[i:last_idx]
                    batch = v[*indices]
                    sentences: list[str] = batch['positive']
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
                    embedding = embedding / embedding.norm(dim=1, keepdim=True).clamp(min=1e-7)
                    if embedding.shape[0] == 1:
                        continue

                    uniformity_loss = lunif(embedding).item()
                    uniformity_losses.append(uniformity_loss)
                    avg_loss: float = sum(uniformity_losses) / len(uniformity_losses)
                    loop.set_postfix(avg_uniformity_loss=avg_loss, current_loss=uniformity_loss)


def main():
    args = parse_args()
    embedder = LossCalculator(args)
    embedder.calculate_losses()


if __name__ == "__main__":
    main()
