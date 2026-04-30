from typing import Any
from model import MessageEmbeddingModel
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch


def parse_args() -> Namespace:
    parser = ArgumentParser("Convert a MessageEmbeddingModel to ONNX")
    parser.add_argument(
        "--model_folder",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_best.pth",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="model.onnx",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--use_dynamo",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args: Namespace = parse_args()
    weights_path: Path = args.model_folder / args.model_name
    train_state_path: Path = args.model_folder / "train_state.pth"
    output_path: Path = args.model_folder / args.output_name

    if not train_state_path.exists():
        print("train_state.pth file must be present in the specified model folder.")
        exit(1)

    train_args: Namespace = torch.load(
        train_state_path,
        weights_only=False,
    )["args"]
    
    try:
        message_context_length: int = train_args.message_context_length
    except AttributeError:
        message_context_length: int = train_args.context_length

    try:
        token_context_length: int = train_args.token_context_length
    except AttributeError:
        token_context_length: int = 512

    model: MessageEmbeddingModel = MessageEmbeddingModel(
        base_model=train_args.base_model,
        message_context_length=message_context_length,
        token_context_length=token_context_length,
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

    weights: dict[str, Any] = torch.load(
        weights_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )["model"]
    model.load_state_dict(weights)
    del weights

    model._base.base = model._base.base.merge_and_unload()
    model.unwrap_base()

    tokens = model.tokenizer(
        [
            "Hello, world",
            "Goodbye, Mars",
        ],
        return_tensors='pt',
        padding=True,
    )
    input_ids = tokens['input_ids']
    token_type_ids = tokens['token_type_ids']
    attention_mask = tokens['attention_mask']

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path.__str__(),
        export_params=True,
        input_names=["input_ids", "attention_mask"],
        output_names=['out_embedding'],
        opset_version=args.opset_version,
        do_constant_folding=True,
        # dynamic_shapes={
        #     'input_ids': {0: 'batch_size', 1: 'seq_len'},
        #     'attention_mask': {0: 'batch_size', 1: 'seq_len'},
        # },
        dynamo=args.use_dynamo,
    )


if __name__ == "__main__":
    main()
