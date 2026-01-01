from data import load_and_split
from datasets import DatasetDict
from model import MessageEmbeddingModel
from vector_database import VectorDatabase
from tqdm import tqdm
from faiss import IndexFlatL2
from pathlib import Path
from argparse import ArgumentParser
import torch


def parse_args():
    parser = ArgumentParser(
        description="Create vector embeddings "
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
        '--output_path',
        type=Path,
        help="Embedding output path",
        required=True,
    )
    return parser.parse_args()


class Embedder:
    def __init__(self, args) -> None:
        self.data_path: Path = args.data_path
        self.model_path: Path = args.model_path
        self.output_path: Path = args.output_path
        self.output_path.mkdir(exist_ok=True, parents=True)
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
        )["val"]

        self.vector_db = VectorDatabase(self.model.embedding_dim)
        self.device = 'cuda'

    def embed_dataset(self):
        current_id = 0
        batch_size: int = 128
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for k, v in self.data.items():
                loop = tqdm(range(0, len(v), batch_size))
                loop.set_description(k)
                for i in loop:
                    last_idx = min(i + batch_size, len(v))
                    sentences: list[str] = v[i:last_idx]['positive']
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
                    for idx in range(last_idx - i):
                        self.vector_db.add(current_id + idx, np_arr[idx])
                    current_id += (last_idx - i)
        self.vector_db.save(self.output_path / 'embeddings.pkl')


def main():
    args = parse_args()
    embedder = Embedder(args)
    embedder.embed_dataset()


if __name__ == "__main__":
    main()
