import atexit
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import math
import os
from pathlib import Path
from typing import Any, Literal, Optional, Union
import warnings

from accelerate import Accelerator
from datasets import DatasetDict
from lion_pytorch import Lion
import mlflow
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from argument_parser import ArgParser
from data import (
    MultipositiveInfoNCEDataset,
    TripletDataset,
    collate_infonce,
    collate_triplet,
    eval_group,
    load_and_split,
)
from loss import clip_loss, infonce_loss, multipositive_infonce_loss, triplet_loss
from lr_scheduling import LinearWarmupCosineDecay
from mlflow_logger import MLFlowLogger, NullLogger
from model import MessageEmbeddingModel


# This should be LossFuncTypeType but that is lame
type LossFuncType = Literal['triplet', 'infonce_multipositive', 'infonce', 'clip']
type PoolingType = Literal['mean', 'attention']


class Trainer:
    def __init__(self) -> None:
        self.init_epoch: int = 1
        self.best_val_loss: float = float('inf')
        self.resuming_training: bool = False
        self.run_id: Optional[str] = None

        self.accelerator: Accelerator
        self.device: torch.device
        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.model: MessageEmbeddingModel
        self.optimizer: optim.Optimizer
        self.lr_scheduler: LRScheduler
        self.lora_config: dict[str, Any]
        self.mlflow_logger: Union[MLFlowLogger, NullLogger]

    def read_args(self, args):
        self.args = args
        self.base_model_name: str = args.base_model
        self.pooling_mode: PoolingType = args.pooling_mode
        self.context_length: int = args.context_length
        self.timestamp: Optional[str] = args.timestamp
        self.loss_func_type: LossFuncType = args.loss_func
        self.loss_func = {
            "triplet": triplet_loss,
            "infonce_multipositive": multipositive_infonce_loss,
            "infonce": infonce_loss,
            "clip": clip_loss,
        }[self.loss_func_type]
        self.use_full_context: bool = args.use_full_context
        self.last_message_only: bool = args.last_message_only
        self.any_message_prob: float = args.any_message_prob
        self.negative_index_distance: Optional[int] = args.negative_index_distance
        self.margin: float = args.margin
        self.temperature: float = args.temperature
        self.lr_ft: float = args.lr_ft
        self.lr_base: float = args.lr_base
        self.lora: bool = args.lora
        self.lora_rank: int = args.lora_rank
        self.lora_alpha: int = args.lora_alpha
        self.lora_dropout: float = args.lora_dropout
        self.mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = args.mixed_precision
        self.out_path: Path = args.out_path
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.epochs: int = args.epochs
        self.optimizer_name: Literal['AdamW', 'Adam', 'Lion'] = args.optimizer_name
        self.weight_decay: float = args.weight_decay
        self.experiment_name: str = args.experiment_name
        self.run_name: str = args.run_name
        self.continue_from: Optional[Path] = None if args.continue_from is None else Path(args.continue_from)
        self.warmup_percentage: float = args.warmup_percentage
        self.lr_scheduler_type: Literal["none", "linear", "lr_warm_cos_dec"] = args.lr_scheduler_type
        self.lr_begin_factor: float = args.lr_begin_factor
        self.lr_end_factor: float = args.lr_end_factor
        self.data_path: Path = args.data_path
        self.train_size: float = args.train_size
        self.batch_size: int = args.batch_size
        self.mlflow_uri: str = args.mlflow_uri
        self.mlflow_username: str = args.mlflow_username
        self.mlflow_password: str = args.mlflow_password
        self.num_workers: int = args.num_workers
        self.gradient_accum_steps: int = args.gradient_accum_steps

        if not self.continue_from:
            self.experiment_path: Path = self.out_path / self.experiment_name
            self.experiment_path.mkdir(exist_ok=True, parents=True)
        self.lora_config: dict[str, Any] = {
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": ["query", "key", "value", "output.dense"],
            "bias": "none",
            "lora_dropout": self.lora_dropout,
        }

    def init_model(self):
        cf: Optional[Path] = self.continue_from
        if cf is None and self.run_name is not None:
            run_path: Path = self.experiment_path / self.run_name
            if (run_path / 'train_state.pth').exists():
                yn = input("A previous run already exists. Continue from that instead? [y/N]:")
                if yn.lower() == 'y':
                    cf = self.experiment_path / self.run_name

        train_state_path: Optional[Path] = cf / "train_state.pth" if cf else None

        if cf is not None:
            assert train_state_path is not None
            self.resuming_training = True

            if train_state_path.exists():
                train_state = torch.load(train_state_path, weights_only=False)
                self.args = train_state["args"]
                self.read_args(self.args)
                self.init_epoch = train_state["epoch"] + 1
                self.run_id = train_state["run_id"]
            else:
                warnings.warn(
                    "No train state file found, training a new model with current config instead!",
                    category=UserWarning,
                )

        self.accelerator = Accelerator(
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accum_steps,
        )
        self.device: torch.device = self.accelerator.device

        if not self.mlflow_uri:
            sqlite_path: Path = self.experiment_path / 'mlflow.db'
            mlflow.set_tracking_uri(f'sqlite:///{sqlite_path.absolute()}')
        else:
            os.environ['MLFLOW_TRACKING_URI'] = self.mlflow_uri
            os.environ['MLFLOW_TRACKING_USERNAME'] = self.mlflow_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = self.mlflow_password

        files: list[Path] = [p for p in self.data_path.glob('*.parquet')]
        dataset: DatasetDict = load_and_split(
            files,
            self.train_size,
            timestamp=datetime.fromisoformat(self.timestamp) if self.timestamp else None
        )

        self.model = MessageEmbeddingModel(
            base_model=self.base_model_name,
            message_context_length=self.context_length,
            pooling_mode=self.pooling_mode,
            use_lora=self.lora,
            lora_config=self.lora_config,
        )
        for k in dataset["train"]:
            dataset["train"][k] = dataset["train"][k].map(eval_group, num_proc=self.num_workers)
        for k in dataset["val"]:
            dataset["val"][k] = dataset["val"][k].map(eval_group, num_proc=self.num_workers)

        if self.loss_func_type == "triplet":
            self.train_dataset = TripletDataset(
                dataset['train'],
                context_len=self.context_length,
                full_context=self.use_full_context,
                last_message_only=self.last_message_only,
                any_message_prob=self.any_message_prob,
                negative_index_distance=self.negative_index_distance,
            )
            self.val_dataset = TripletDataset(
                dataset['val'],
                context_len=self.context_length,
                full_context=self.use_full_context,
                last_message_only=self.last_message_only,
                any_message_prob=self.any_message_prob,
                negative_index_distance=self.negative_index_distance,
            )
        elif self.loss_func_type == "infonce_multipositive":
            self.train_dataset = MultipositiveInfoNCEDataset(
                dataset['train'],
                context_len=self.context_length,
            )
            self.val_dataset = MultipositiveInfoNCEDataset(
                dataset['val'],
                context_len=self.context_length,
            )
        elif self.loss_func_type == "infonce":
            self.train_dataset = TripletDataset(
                dataset['train'],
                context_len=self.context_length,
                full_context=self.use_full_context,
                last_message_only=self.last_message_only,
                any_message_prob=self.any_message_prob,
                negative_index_distance=self.negative_index_distance,
                no_negatives=True,
            )
            self.val_dataset = TripletDataset(
                dataset['val'],
                context_len=self.context_length,
                full_context=self.use_full_context,
                last_message_only=self.last_message_only,
                any_message_prob=self.any_message_prob,
                negative_index_distance=self.negative_index_distance,
                no_negatives=True,
            )
        elif self.loss_func_type == "clip":
            self.train_dataset = TripletDataset(
                dataset['train'],
                context_len=self.context_length,
                full_context=self.use_full_context,
                last_message_only=self.last_message_only,
                any_message_prob=self.any_message_prob,
                negative_index_distance=self.negative_index_distance,
                no_negatives=True,
            )
            self.val_dataset = TripletDataset(
                dataset['val'],
                context_len=self.context_length,
                full_context=self.use_full_context,
                last_message_only=self.last_message_only,
                any_message_prob=self.any_message_prob,
                negative_index_distance=self.negative_index_distance,
                no_negatives=True,
            )
        else:
            raise AssertionError("Unreachable")
        self.optimizer = self.get_new_optimizer()
        steps_per_epoch: int = math.ceil(len(self.train_dataset) / self.batch_size)
        total_steps: int = self.epochs * steps_per_epoch
        warmup_steps: int = int(self.warmup_percentage * total_steps)

        match self.lr_scheduler_type:
            case "none":
                self.lr_scheduler = LambdaLR(
                    self.optimizer,
                    lambda _: 1.0
                )
            case "linear":
                self.lr_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=self.lr_end_factor,
                    total_iters=total_steps,
                )
            case "lr_warm_cos_dec":
                self.lr_scheduler = LinearWarmupCosineDecay(
                    self.optimizer,
                    start_factor=self.lr_begin_factor,
                    end_factor=self.lr_end_factor,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                )

        if cf is not None:
            assert train_state_path is not None

            if train_state_path.exists():
                train_state = torch.load(train_state_path, weights_only=False)
                self.optimizer.load_state_dict(train_state["optimizer"])
                self.lr_scheduler.load_state_dict(train_state["lr_scheduler"])

                last_model_path: Path = cf / "model_last.pth"
                best_model_path: Path = cf / "model_best.pth"

                if last_model_path.exists():
                    model_state: dict[str, Any] = torch.load(last_model_path)
                    self.model.load_state_dict(model_state["model"])
                    self.best_val_loss: float = model_state["val_loss"]  # Fallback if model_best.pth does not exists
                else:
                    warnings.warn(
                        "No last model found! Training from random weights from the first epoch instead!",
                        category=UserWarning,
                    )
                    self.init_epoch = 1

                if best_model_path.exists():
                    self.best_val_loss: float = torch.load(best_model_path)["val_loss"]

    def get_new_optimizer(self) -> optim.Optimizer:
        param_groups: dict[str, Any] = self.model.get_param_groups()
        optim_input: list[dict[str, Any]] = [
            {
                "params": param_groups['base'],
                "lr": self.lr_ft,
                "weight_decay": self.weight_decay,
            },
            {
                "params": param_groups['additional'],
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
        if self.accelerator.is_main_process:
            mlflow.set_experiment(self.experiment_name)
            extra_run_kwargs = {}
            if self.run_name:
                extra_run_kwargs |= {'run_name': self.run_name}
            ctx = mlflow.start_run(log_system_metrics=True, run_id=self.run_id, **extra_run_kwargs)
            self.mlflow_logger = MLFlowLogger(self.run_id)
        else:
            ctx = nullcontext()
            self.mlflow_logger = NullLogger()

        with ctx as run:
            self.run_id = run.info.run_id if run else self.run_id
            self.mlflow_logger.run_id = self.run_id

            run_path: Path = Path()
            if self.accelerator.is_main_process:
                self.run_name = self.run_name or run.data.tags.get("mlflow.runName")
                run_path: Path = self.experiment_path / self.run_name
                is_empty: bool = not run_path.exists() or not any(run_path.iterdir())
                run_path.mkdir(exist_ok=self.resuming_training or is_empty)

                total_params: int = sum([p.numel() for p in self.model.parameters()])
                print(f"Model has {total_params} parameters.")
                self.mlflow_logger.log_param("lr_ft", self.lr_ft)
                self.mlflow_logger.log_param("lr_base", self.lr_base)
                self.mlflow_logger.log_param("weight_decay", self.weight_decay)
                self.mlflow_logger.log_param("base_model_name", self.base_model_name)
                self.mlflow_logger.log_param("pooling_mode", self.pooling_mode)
                self.mlflow_logger.log_param("context_length", self.context_length)
                self.mlflow_logger.log_param("margin", self.margin)
                self.mlflow_logger.log_param("lora", self.lora)
                self.mlflow_logger.log_param("lora_rank", self.lora_rank)
                self.mlflow_logger.log_param("lora_alpha", self.lora_alpha)
                self.mlflow_logger.log_param("lora_dropout", self.lora_dropout)
                self.mlflow_logger.log_param("optimizer_name", self.optimizer_name)
                self.mlflow_logger.log_param("lr_end_factor", self.lr_end_factor)
                self.mlflow_logger.log_param("train_size", self.train_size)
                self.mlflow_logger.log_param("batch_size", self.batch_size)
                self.mlflow_logger.log_param("mixed_precision", self.mixed_precision)
                self.mlflow_logger.log_param("param_count", total_params)

            if self.loss_func_type == 'triplet':
                collate_fn = collate_triplet
            elif self.loss_func_type == 'infonce_multipositive':
                collate_fn = collate_infonce
            elif self.loss_func_type == 'infonce':
                collate_fn = collate_infonce
            elif self.loss_func_type == 'clip':
                collate_fn = collate_infonce
            else:
                raise AssertionError("Unreachable")
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            self.model, self.optimizer, train_loader, val_loader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, train_loader, val_loader, self.lr_scheduler
            )

            token_context_length: int = self.model.token_context_length

            if self.loss_func_type == 'triplet':
                one_step_func = self._one_step_triplet
            elif self.loss_func_type == 'infonce_multipositive':
                one_step_func = self._one_step_multipos_infonce
            elif self.loss_func_type == 'infonce':
                one_step_func = partial(self._one_step_infonce_clip, clip=False)
            elif self.loss_func_type == 'clip':
                one_step_func = partial(self._one_step_infonce_clip, clip=True)
            else:
                raise AssertionError("Unreachable")

            for epoch in range(self.init_epoch, self.epochs + 1):
                # ---------------------------------
                # Train
                # ---------------------------------
                if self.accelerator.is_main_process:
                    train_loop = tqdm(train_loader)
                    train_loop.set_description(f"Training [{epoch}/{self.epochs}]")
                else:
                    train_loop = train_loader

                total_train_loss: float = 0.0
                train_losses: list[float] = []

                train_iters = 0
                self.model.train()
                for train_batch in train_loop:
                    current_lr: float = self.optimizer.param_groups[0]["lr"]
                    train_iters += 1

                    if self.loss_func_type == 'triplet':
                        batch = {
                            "anchors": train_batch[0],
                            "positives": train_batch[1],
                            "negatives": train_batch[2],
                        }
                    elif self.loss_func_type == 'infonce_multipositive':
                        batch = {
                            "anchors": train_batch[0],
                            "positives": train_batch[1],
                        }
                    elif self.loss_func_type == 'infonce':
                        batch = {
                            "anchors": train_batch[0],
                            "positives": train_batch[1],
                        }
                    elif self.loss_func_type == 'clip':
                        batch = {
                            "anchors": train_batch[0],
                            "positives": train_batch[1],
                        }
                    else:
                        raise AssertionError("Unreachable")

                    with self.accelerator.accumulate(self.model):
                        self.optimizer.zero_grad()
                        loss = one_step_func(
                            batch,
                            token_context_length,
                            margin=self.margin,
                            temperature=self.temperature,
                        )
                        train_losses.append(loss.item())
                        total_train_loss += loss.item()
                        avg_loss: float = total_train_loss / train_iters

                        if self.accelerator.is_main_process:
                            num_last: int = 100
                            last_avg = sum(train_losses[-num_last:]) / train_losses[-num_last:].__len__()
                            train_loop.set_postfix(loss=loss.item(), avg_loss=avg_loss, last_100_avg=last_avg, lr=current_lr)

                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        self.lr_scheduler.step()

                if self.accelerator.is_main_process:
                    losses_tensor: torch.Tensor = torch.tensor(train_losses, device=self.accelerator.device)
                    losses_gathered: torch.Tensor = self.accelerator.gather_for_metrics(losses_tensor)
                    avg_train_loss: float = losses_gathered.mean().item()
                    self.mlflow_logger.log_metric("train_loss", avg_train_loss, step=epoch)

                # ---------------------------------
                # Validation
                # ---------------------------------
                if self.accelerator.is_main_process:
                    val_loop = tqdm(val_loader)
                    val_loop.set_description(f"Validation [{epoch}/{self.epochs}]")
                else:
                    val_loop = val_loader

                total_val_loss: float = 0.0
                avg_val_loss: float = float('inf')
                val_losses: list[float] = []

                self.model.eval()
                with torch.no_grad():
                    val_iters = 0
                    for train_batch in val_loop:
                        val_iters += 1

                        if self.loss_func_type == 'triplet':
                            batch = {
                                "anchors": train_batch[0],
                                "positives": train_batch[1],
                                "negatives": train_batch[2],
                            }
                        elif self.loss_func_type == 'infonce_multipositive':
                            batch = {
                                "anchors": train_batch[0],
                                "positives": train_batch[1],
                            }
                        elif self.loss_func_type == 'infonce':
                            batch = {
                                "anchors": train_batch[0],
                                "positives": train_batch[1],
                            }
                        elif self.loss_func_type == 'clip':
                            batch = {
                                "anchors": train_batch[0],
                                "positives": train_batch[1],
                            }
                        else:
                            raise AssertionError("Unreachable")

                        loss = one_step_func(
                            batch,
                            token_context_length,
                            margin=self.margin,
                            temperature=self.temperature,
                        )
                        total_val_loss += loss.item()
                        val_losses.append(loss.item())
                        avg_loss: float = total_val_loss / val_iters
                        if self.accelerator.is_main_process:
                            num_last: int = 100
                            last_avg = sum(val_losses[-num_last:]) / val_losses[-num_last:].__len__()
                            val_loop.set_postfix(loss=loss.item(), last_100_avg=last_avg, avg_loss=avg_loss)

                    if self.accelerator.is_main_process:
                        val_losses_tensor: torch.Tensor = torch.tensor(val_losses, device=self.accelerator.device)
                        val_losses_gathered: torch.Tensor = self.accelerator.gather_for_metrics(val_losses_tensor)
                        avg_val_loss: float = val_losses_gathered.mean().item()
                        self.mlflow_logger.log_metric("val_loss", avg_val_loss, step=epoch)

                # ---------------------------------
                # Model saving
                # ---------------------------------
                self.save_model(
                    run_path=run_path,
                    epoch=epoch,
                    val_loss=avg_val_loss,
                )
                self.best_val_loss = min(self.best_val_loss, avg_val_loss)
                self.accelerator.wait_for_everyone()

    def _one_step_triplet(
        self,
        batch: dict[str, list[str]],
        max_length: int,
        margin: float,
        **kwargs,
    ) -> torch.Tensor:
        inputs: dict[str, dict[str, Any]] = {
            k: self.model.tokenizer(
                v,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            for k, v in batch.items()
        }
        inputs = {
            k: {
                kk: vv.to(self.device)
                for kk, vv in v.items()
            }
            for k, v in inputs.items()
        }

        outputs = {
            k: self.model(**v)
            for k, v in inputs.items()
        }
        loss = triplet_loss(
            **outputs,
            margin=margin,
        )

        return loss

    def _one_step_multipos_infonce(
        self,
        batch: dict[str, Any],
        max_length: int,
        temperature: float,
        **kwargs,
    ) -> torch.Tensor:
        batch["positives"] = [s for sub in batch["positives"] for s in sub]
        inputs: dict[str, dict[str, Any]] = {
            "anchors": self.model.tokenizer(
                batch["anchors"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            ),
            "positives": self.model.tokenizer(
                batch["positives"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        }
        inputs = {
            k: {
                kk: vv.to(self.device)
                for kk, vv in v.items()
            }
            for k, v in inputs.items()
        }

        outputs = {
            k: self.model(**v)
            for k, v in inputs.items()
        }
        loss = multipositive_infonce_loss(
            **outputs,
            temperature=temperature,
        )

        return loss

    def _one_step_infonce_clip(
        self,
        batch: dict[str, Any],
        max_length: int,
        temperature: float,
        clip: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        batch["positives"] = [s for sub in batch["positives"] for s in sub]
        inputs: dict[str, dict[str, Any]] = {
            "anchors": self.model.tokenizer(
                batch["anchors"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            ),
            "positives": self.model.tokenizer(
                batch["positives"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        }
        inputs = {
            k: {
                kk: vv.to(self.device)
                for kk, vv in v.items()
            }
            for k, v in inputs.items()
        }

        outputs = {
            k: self.model(**v)
            for k, v in inputs.items()
        }
        loss_fn = clip_loss if clip else infonce_loss
        loss = loss_fn(
            **outputs,
            temperature=temperature,
        )

        return loss

    def save_model(
        self,
        run_path: Path,
        epoch: int,
        val_loss: float,
    ):
        if not self.accelerator.is_main_process:
            return

        unwrapped_model: MessageEmbeddingModel = self.accelerator.unwrap_model(self.model)
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'args': self.args,
            'epoch': epoch,
            'run_id': self.run_id,
        }, run_path / "train_state.pth")
        torch.save({
            'model': unwrapped_model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
        }, run_path / "model_last.pth")

        if val_loss < self.best_val_loss:
            torch.save({
                'model': self.model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, run_path / "model_best.pth")


def main():
    parser: ArgParser = ArgParser()
    args = parser.parse_args()
    if not args.continue_from:
        if not args.base_model:
            parser.error("--base_model is required.")
        if not args.experiment_name:
            parser.error("--experiment_name is required.")

    trainer = Trainer()
    trainer.read_args(args)
    trainer.init_model()
    try:
        trainer.train()
    finally:
        trainer.mlflow_logger.stop()


if __name__ == '__main__':
    main()
