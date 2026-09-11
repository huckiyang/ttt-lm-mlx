"""Convert local reference PyTorch/Hugging Face weights to an MLX checkpoint."""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

from .config import ModelArgs
from .io import save_model
from .model import Model


def convert(source, output, dtype="float32"):
    source, output = Path(source), Path(output)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "ttt":
        raise ValueError("Only the reference TTT architecture is supported")
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Conversion expects unquantized reference weights")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported conversion dtype")
    args = ModelArgs.from_dict(config)
    model = Model(args)
    shards = sorted(source.glob("model*.safetensors"))
    weights = {}
    if shards:
        for shard in shards:
            current = mx.load(str(shard))
            if weights.keys() & current.keys():
                raise ValueError("Duplicate weight keys across shards")
            weights.update(current)
    else:
        import torch

        shards = sorted(source.glob("pytorch_model*.bin"))
        if not shards:
            raise FileNotFoundError(
                "No model*.safetensors or pytorch_model*.bin weights"
            )
        for shard in shards:
            current = torch.load(shard, map_location="cpu", weights_only=True)
            if weights.keys() & current.keys():
                raise ValueError("Duplicate weight keys across shards")
            weights.update({k: mx.array(v.float().numpy()) for k, v in current.items()})
    target_dtype = getattr(mx, dtype)
    model.load_weights(
        [(k, v.astype(target_dtype)) for k, v in model.sanitize(weights).items()]
    )
    # Keep learned fast-weight initializers and update-rate scalars in FP32.
    for index, layer in enumerate(model.layers):
        ttt = layer.seq_modeling_block
        for name in (
            *ttt.fast_weight_names,
            "learnable_token_idx",
            "learnable_ttt_lr_weight",
            "learnable_ttt_lr_bias",
            "ttt_norm_weight",
            "ttt_norm_bias",
        ):
            key = f"model.layers.{index}.seq_modeling_block.{name}"
            setattr(ttt, name, weights[key].astype(mx.float32))
    mx.eval(model.parameters())
    save_model(output, model, config=config)
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "chat_template.jinja",
    ):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    args = parser.parse_args()
    convert(args.source, args.output, args.dtype)
    print(f"Saved MLX TTT checkpoint to {args.output}")


if __name__ == "__main__":
    main()
