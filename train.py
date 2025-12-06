from argparse import ArgumentParser
from contextlib import nullcontext
import math
import os
from pathlib import Path
from typing import Literal, Optional

from accelerate import Accelerator
from datasets import DatasetDict
from lion_pytorch import Lion
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MessageEmbeddingModel
from triplet_dataset import TripletDataset, collate_triplet, load_and_split


class Trainer:
    def __init__(self) -> None:
        self.accelerator = Accelerator()
        self.device: torch.device = self.accelerator.device

        self.create_argparser()
        args = self.parser.parse_args()
        self.base_model_name: str = args.base_model
        self.pooling_mode: Literal['mean', 'attention'] = args.pooling_mode
        self.context_length: int = args.context_length
        self.icp: float = args.icp
        self.margin: float = args.margin
        self.alpha: float = args.alpha
        self.lr_ft: float = args.lr_ft
        self.lr_base: float = args.lr_base
        self.lora: bool = args.lora
        self.out_path: Path = args.out_path
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.epochs: int = args.epochs
        self.optimizer_name: Literal['AdamW', 'Adam', 'Lion'] = args.optimizer_name
        self.weight_decay: float = args.weight_decay
        self.experiment_name: str = args.experiment_name
        self.run_name: str = args.run_name
        self.continue_from: Optional[Path] = None if not args.continue_from else Path(args.continue_from)
        self.lr_end_factor: float = args.lr_end_factor
        self.data_path: Path = args.data_path
        self.train_size: float = args.train_size
        self.batch_size: int = args.batch_size
        self.mlflow_uri: str = args.mlflow_uri
        self.mlflow_username: str = args.mlflow_username
        self.mlflow_password: str = args.mlflow_password

        self.experiment_path: Path = self.out_path / self.experiment_name
        self.experiment_path.mkdir(exist_ok=True, parents=True)

        self.train_dataset: TripletDataset
        self.val_dataset: TripletDataset
        self.model: MessageEmbeddingModel
        self.optimizer: optim.Optimizer
        self.lr_scheduler: LRScheduler

    def init_model(self):
        sqlite_path: Path = self.experiment_path / 'mlflow.db'
        if not self.mlflow_uri:
            mlflow.set_tracking_uri(f'sqlite:///{sqlite_path.absolute()}')
        else:
            os.environ['MLFLOW_TRACKING_URI'] = self.mlflow_uri
            os.environ['MLFLOW_TRACKING_USERNAME'] = self.mlflow_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = self.mlflow_password

        files: list[Path] = [p for p in self.data_path.glob('*.parquet')]
        dataset: DatasetDict = load_and_split(files, self.train_size)
        self.train_dataset = TripletDataset(
            dataset['train'],
            context_len=self.context_length,
            in_context_probability=self.icp,
            base_margin=self.margin,
            alpha=self.alpha,
        )
        self.val_dataset = TripletDataset(
            dataset['val'],
            context_len=self.context_length,
            in_context_probability=self.icp,
            base_margin=self.margin,
            alpha=self.alpha,
        )
        self.model = MessageEmbeddingModel(
            base_model=self.base_model_name,
            pooling_mode=self.pooling_mode,
            message_context_length=self.context_length,
        )
        self.optimizer = self.get_new_optimizer()
        steps_per_epoch: int = math.ceil(len(self.train_dataset) / self.batch_size)
        self.lr_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.lr_end_factor,
            total_iters=self.epochs * steps_per_epoch
        )

    def create_argparser(self) -> None:
        self.parser: ArgumentParser = ArgumentParser(
            description="Fine-tuning script for Botphy's memories extension."
        )
        self.parser.add_argument(
            '--base_model',
            type=str,
            required=True,
            choices=[
                'TURKCELL/roberta-base-turkish-uncased',
                'dbmdz/bert-base-turkish-128k-uncased',
                'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr',
            ],
            help="Model name to finetune.",
        )
        self.parser.add_argument(
            '--pooling_mode',
            type=str,
            choices=[
                'mean',
                'attention',
            ],
            default='mean',
            help="Pooling strategy",
        )
        self.parser.add_argument(
            '--context_length',
            type=int,
            default=8,
            help='Window size for when sampling messages'
        )
        self.parser.add_argument(
            '--icp', '--in_context_probability',
            type=float,
            default=0.05,
            help="Probability of sampling the negative sentece within the context.",
        )
        self.parser.add_argument(
            '--margin',
            type=float,
            default=1.0,
            help="Margin value. Ensures the positive sentence is this amount of closer to negative sentence by this amount. Please refer to https://arxiv.org/abs/1908.10084 section 3 for more details.",  # noqa
        )
        self.parser.add_argument(
            '--alpha',
            type=float,
            default=1.0,
            help="Used when calculating the margin of negative sentences if they happen to be in context. Check triplet_dataset.py for more details.",  # noqa
        )
        self.parser.add_argument(
            '--lr_ft',
            type=float,
            default=1e-5,
            help="Learning rate for the fine-tuned parameters.",
        )
        self.parser.add_argument(
            '--lr_base',
            type=float,
            default=1e-4,
            help="Learning rate for the new parameters.",
        )
        self.parser.add_argument(
            '--lora',
            action='store_true',
            help="Use LoRa (https://arxiv.org/abs/2106.09685) for fine-tuning.",
        )
        self.parser.add_argument(
            '--out_path',
            type=Path,
            default=Path('./results'),
            help="Path to save outputs.",
        )
        self.parser.add_argument(
            '--epochs',
            type=int,
            default=20,
            help="Number of epochs.",
        )
        self.parser.add_argument(
            '--optimizer_name',
            type=str,
            default='AdamW',
            choices=[
                'AdamW',
                'Adam',
                'Lion',
            ],
            help="Optimizer to use.",
        )
        self.parser.add_argument(
            '--weight_decay',
            type=float,
            default=0.01,
            help="Weight decay value.",
        )
        self.parser.add_argument(
            '--experiment_name',
            type=str,
            required=True,
            help="The name of the experiment.",
        )
        self.parser.add_argument(
            '--run_name',
            type=str,
            default=None,
            help="The name of the MLFlow run."
        )
        self.parser.add_argument(
            '--continue_from',
            type=str,
            default='',
            help="Path to continue training from."
        )
        self.parser.add_argument(
            '--lr_end_factor',
            type=float,
            default=0.1,
            help="End factor of the LinearLR"
        )
        self.parser.add_argument(
            '--data_path',
            type=Path,
            default=Path('./data/'),
            help="Input data for the model."
        )
        self.parser.add_argument(
            '--train_size',
            type=float,
            default=0.85,
            help="Percentage size of the train split, the remainder is used for validation."
        )
        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=16,
            help="Batch size used for training",
        )
        self.parser.add_argument(
            '--mlflow_uri',
            type=str,
            default='',
            help="MLFlow URI. An SQLite database is used if not provided."
        )
        self.parser.add_argument(
            "--mlflow_username",
            type=str,
            default='',
            help="MLFlow remote server username."
        )
        self.parser.add_argument(
            "--mlflow_password",
            type=str,
            default='',
            help="MLFlow remote server password."
        )

    def get_new_optimizer(self) -> optim.Optimizer:
        param_groups: dict[str, Any] = self.model.get_param_groups()
        optim_input: list[dict[str, Any]] = [
            {
                "params": param_groups['old'],
                "lr": self.lr_ft,
                "weight_decay": self.weight_decay,
            },
            {
                "params": param_groups['new'],
                'lr': self.lr_base,
                "weight_decay": self.weight_decay,
            }
        ]
        if self.optimizer_name == 'Adam':
            return optim.Adam(optim_input)
        elif self.optimizer_name == 'AdamW':
            return optim.AdamW(optim_input)
        elif self.optimizer_name == 'Lion':
            return Lion(optim_input)
        else:
            raise NotImplementedError()

    def train(self) -> None:
        # TODO: Load train state if continue_from is given
        # TODO: Implement LoRA

        if self.accelerator.is_main_process:
            mlflow.set_experiment(self.experiment_name)
            extra_run_kwargs = {}
            if self.run_name:
                extra_run_kwargs = {'run_name': self.run_name}
            ctx = mlflow.start_run(log_system_metrics=True, **extra_run_kwargs)
        else:
            ctx = nullcontext()

        with ctx as run:
            if self.accelerator.is_main_process:
                self.run_name = self.run_name or run.data.tags.get("mlflow.runName")
                run_path: Path = self.experiment_path / self.run_name
                run_path.mkdir(exist_ok=False)

                total_params: int = sum([p.numel() for p in self.model.parameters()])
                print(f"Model has {total_params} parameters.")
                mlflow.log_param("lr_ft", self.lr_ft)
                mlflow.log_param("lr_base", self.lr_base)
                mlflow.log_param("weight_decay", self.weight_decay)
                mlflow.log_param("base_model_name", self.base_model_name)
                mlflow.log_param("pooling_mode", self.pooling_mode)
                mlflow.log_param("context_length", self.context_length)
                mlflow.log_param("in_context_probability", self.icp)
                mlflow.log_param("margin", self.margin)
                mlflow.log_param("alpha", self.alpha)
                mlflow.log_param("lora", self.lora)
                mlflow.log_param("optimizer_name", self.optimizer_name)
                mlflow.log_param("lr_end_factor", self.lr_end_factor)
                mlflow.log_param("train_size", self.train_size)
                mlflow.log_param("batch_size", self.batch_size)
                mlflow.log_param("param_count", total_params)

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_triplet
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_triplet
            )
            self.model, self.optimizer, train_loader, val_loader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, train_loader, val_loader, self.lr_scheduler
            )

            best_val_loss: float = float('inf')
            tokenizer = self.model.tokenizer
            token_context_length: int = self.model.token_context_length
            for epoch in range(1, self.epochs + 1):
                total_train_loss = 0.0
                total_val_loss = 0.0

                if self.accelerator.is_main_process:
                    train_loop = tqdm(train_loader)
                    train_loop.set_description(f"Training [{epoch}/{self.epochs}]")
                else:
                    train_loop = train_loader

                train_iters = 0
                self.model.train()
                for anchors, positives, negatives, margins in train_loop:
                    current_lr: float = self.optimizer.param_groups[0]["lr"]
                    train_iters += 1

                    self.optimizer.zero_grad()
                    loss = self._one_step(anchors, positives, negatives, margins, tokenizer, token_context_length)
                    total_train_loss += loss.item()
                    avg_loss: float = total_train_loss / train_iters

                    if self.accelerator.is_main_process:
                        train_loop.set_postfix(loss=loss.item(), avg_loss=avg_loss, lr=current_lr)

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                avg_train_loss: float = total_train_loss / train_iters
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

                # ---------------------------------
                # Validation
                # ---------------------------------
                if self.accelerator.is_main_process:
                    val_loop = tqdm(val_loader)
                    val_loop.set_description(f"Validation [{epoch}/{self.epochs}]")
                else:
                    val_loop = val_loader

                self.model.eval()
                with torch.no_grad():
                    val_iters = 0
                    for anchors, positives, negatives, margins in val_loop:
                        val_iters += 1
                        loss = self._one_step(anchors, positives, negatives, margins, tokenizer, token_context_length)
                        total_val_loss += loss.item()
                        avg_loss: float = total_val_loss / val_iters
                        if self.accelerator.is_main_process:
                            val_loop.set_postfix(loss=loss.item(), avg_loss=avg_loss)

                    avg_val_loss: float = total_val_loss / val_iters
                    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

                # Model saving
                if self.accelerator.is_main_process:
                    torch.save({
                        'optimizer': self.optimizer.state_dict(),
                        'lr_scheduler': self.optimizer.state_dict(),
                        'epoch': epoch
                    }, run_path / "train_state.pth")
                    torch.save({
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                    }, run_path / "model_last.pth")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save({
                            'model': self.model.state_dict(),
                            'epoch': epoch,
                        }, run_path / "model_best.pth")

                self.accelerator.wait_for_everyone()

    def _one_step(
        self,
        anchors: list[str],
        positives: list[str],
        negatives: list[str],
        margins: list[float],
        tokenizer,
        max_length: int,
    ) -> torch.Tensor:
        anchor_in = tokenizer(
            anchors,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        positive_in = tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        negative_in = tokenizer(
            negatives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        anchor_tok = anchor_in['input_ids'].to(self.device)
        anchor_mask = anchor_in['attention_mask'].to(self.device)
        positive_tok = positive_in['input_ids'].to(self.device)
        positive_mask = positive_in['attention_mask'].to(self.device)
        negative_tok = negative_in['input_ids'].to(self.device)
        negative_mask = negative_in['attention_mask'].to(self.device)

        anchor_out = self.model(anchor_tok, anchor_mask)
        pos_out = self.model(positive_tok, positive_mask)
        neg_out = self.model(negative_tok, negative_mask)

        d_ap = torch.linalg.norm(anchor_out - pos_out, dim=-1)
        d_an = torch.linalg.norm(anchor_out - neg_out, dim=-1)
        loss = F.relu(d_ap - d_an + margins)
        loss = torch.mean(loss, dim=0)

        return loss


def main():
    trainer = Trainer()
    trainer.init_model()
    trainer.train()


if __name__ == '__main__':
    main()
