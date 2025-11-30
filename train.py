import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional


class Trainer:
    def __init__(self) -> None:
        self.create_argparser()

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
            type=bool,
            action='store_true',
            help="Use LoRa (https://arxiv.org/abs/2106.09685) for fine-tuning.",
        )
        self.parser.add_argument(
            '--out_path',
            type=Path,
            default='./results',
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
            help="Weight decay value. Unnecessary if Adam optimizer is used.",
        )
        self.parser.add_argument(
            '--experiment_name',
            type=str,
            required=True,
            help="The name of the experiment.",
        )
        self.parser.add_argument(
            '--continue_from',
            type=str,
            default='',
            help="Path to continue training from."
        )
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
        self.out_path: Path = Path(args.out_path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.epochs: int = args.epochs
        self.optimizer_name: Literal['AdamW', 'Adam', 'Lion'] = args.optimizer_name
        self.weight_decay: float = args.weight_decay
        self.experiment_name: str = args.experiment_name
        self.continue_from: Optional[Path] = None if not args.continue_from else Path(args.continue_from)

    def train(self) -> None:
        pass


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
