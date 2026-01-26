from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import NoReturn


class ArgParser:
    def __init__(self):
        self.parser: ArgumentParser = ArgumentParser(
            description="Fine-tuning script for Botphy's memories extension."
        )
        self.parser.add_argument(
            '--base_model',
            type=str,
            # required=True,
            choices=[
                'TURKCELL/roberta-base-turkish-uncased',
                'dbmdz/bert-base-turkish-128k-uncased',
                'emrecan/bert-base-turkish-cased-mean-nli-stsb-tr',
            ],
            default=None,
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
            '--timestamp',
            type=str,
            default=None,
            help="A timestamp in ISO-8601 format. Used to include the data entries AFTER this timestamp.",
        )
        self.parser.add_argument(
            '--loss_func',
            type=str,
            choices=[
                'triplet',
                'infonce_multipositive',
                'infonce',
                'clip'
            ],
            default="triplet",
            help="Loss function to use."
        )
        self.parser.add_argument(
            '--use_full_context',
            default=False,
            action='store_true',
            help="For Triplet Loss only. Returns all sub-messages of a block as its anchor. Effectively increases the train and validation dataset size 8 fold.",  # noqa
        )
        self.parser.add_argument(
            '--last_message_only',
            default=False,
            action='store_true',
            help="For Triplet Loss only. Return only the last message of a block as its anchor. Ignored if --use_full_context is provided.",  # noqa
        )
        self.parser.add_argument(
            '--any_message_prob',
            type=float,
            default=0.1,
            help="Probability to sample any message of current block even though last_message_only is true. Used for regularization."  # noqa
        )
        self.parser.add_argument(
            '--negative_index_distance',
            default=None,
            type=int,
            help="The maximum index distance of the negative example from the positive. For hard negatives."
        )
        self.parser.add_argument(
            '--margin',
            type=float,
            default=0.3,
            help="Margin value. Ensures the positive sentence is this amount of closer to negative sentence by this amount. Please refer to https://arxiv.org/abs/1908.10084 section 3 for more details.",  # noqa
        )
        self.parser.add_argument(
            '--temperature',
            type=float,
            default=0.15,
            help="Temperature value for InfoNCE or CLIP loss functions.",
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
            "--lora_rank",
            type=int,
            default=8,
            help="Rank parameter of LoRA.",
        )
        self.parser.add_argument(
            "--lora_alpha",
            type=int,
            default=16,
            help="Alpha parameter of LoRA. It is recommended to keep alpha/rank \\in O(1)",
        )
        self.parser.add_argument(
            "--lora_dropout",
            type=float,
            default=0.05,
            help="Dropout parameter of LoRA.",
        )
        self.parser.add_argument(
            "--mixed_precision", "--mp",
            type=str,
            default="no",
            choices=["no", "fp16", "bf16", "fp8"],
            help="The mixed precision type to use.",
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
            default=None,
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
            default=None,
            help="Path to continue training from."
        )
        self.parser.add_argument(
            '--warmup_percentage',
            type=float,
            default=0.05,
            help="Warmup percentage for linear warmup + cos decay scheduler. Unused otherwise."
        )
        self.parser.add_argument(
            '--lr_scheduler_type',
            type=str,
            choices=["none", "linear", "lr_warm_cos_dec"],
            default="linear",
            help="Learning rate scheduler type.",
        )
        self.parser.add_argument(
            '--lr_begin_factor',
            type=float,
            default=0.1,
            help="Start factor for linear warmup + cos decay scheduler. Unused if LinearLR is used."
        )
        self.parser.add_argument(
            '--lr_end_factor',
            type=float,
            default=0.01,
            help="End factor of the LinearLR or linear warmup + cos decay schedulers."
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
            '--mlflow_username',
            type=str,
            default='',
            help="MLFlow remote server username."
        )
        self.parser.add_argument(
            '--mlflow_password',
            type=str,
            default='',
            help="MLFlow remote server password."
        )
        self.parser.add_argument(
            '--num_workers',
            type=int,
            default=4,
            help="Number of workers for the dataloader."
        )
        self.parser.add_argument(
            '--gradient_accum_steps',
            type=int,
            default=1,
            help="Gradient accumulation steps."
        )

    def parse_args(self) -> Namespace:
        return self.parser.parse_args()

    def error(self, message) -> NoReturn:
        self.parser.error(message)
