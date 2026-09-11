"""Local checkpoint I/O through MLX-LM's public model-class hook."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model as mlx_load_model, load_tokenizer

from .config import ModelArgs
from .model import Model


def model_classes(config):
    if config.get("model_type") != "ttt":
        raise ValueError("Expected a TTT checkpoint (model_type=ttt)")
    return Model, ModelArgs


def load_model(path, lazy=False):
    path = Path(path)
    model, config = mlx_load_model(path, lazy=lazy, get_model_classes=model_classes)
    # Keep JSON metadata outside the nn.Module array/state tree.
    object.__setattr__(model, "_checkpoint_config", deepcopy(config))
    generation_path = path / "generation_config.json"
    if generation_path.exists():
        object.__setattr__(
            model,
            "_checkpoint_generation_config",
            json.loads(generation_path.read_text()),
        )
    return model, config


def load(path, tokenizer_config=None, lazy=False):
    model, config = load_model(path, lazy=lazy)
    tokenizer = load_tokenizer(
        Path(path), tokenizer_config or {}, eos_token_ids=config.get("eos_token_id")
    )
    return model, tokenizer


def _save_config(model, tokenizer, config):
    result = deepcopy(getattr(model, "_checkpoint_config", {}))
    if config is not None:
        result.update(deepcopy(config))
    result.update(asdict(model.args))
    # These checkpoints load through our model-class hook.
    result.pop("model_file", None)

    quantized = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.QQLinear):
            raise ValueError(
                "Saving activation-quantized QQLinear modules is not supported"
            )
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            quantized[name] = {
                "group_size": module.group_size,
                "bits": module.bits,
                "mode": module.mode,
            }
    # Rebuild from current modules so partial quantization and dequantization
    # cannot leave stale checkpoint settings behind.
    for name in ("quantization", "quantization_config", "quantize_activations"):
        result.pop(name, None)
    if quantized:
        result["quantization"] = {**next(iter(quantized.values())), **quantized}

    if tokenizer is not None:
        if isinstance(tokenizer, TokenizerWrapper):
            result["eos_token_id"] = sorted(
                i for i in tokenizer.eos_token_ids if i is not None
            )
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if result.get(name) is None:
                value = getattr(tokenizer, name, None)
                if value is not None:
                    result[name] = value
    return result


def save_model(path, model, tokenizer=None, *, config=None):
    """Save native weights and generation metadata, including weight quantization.

    Loaded metadata is retained automatically. ``config`` can supply additional
    metadata for a new model; architecture and quantization follow the model.
    """
    path = Path(path)
    if (path / "config.json").exists() or list(path.glob("model*.safetensors")):
        raise FileExistsError(f"Checkpoint already exists in {path}")
    # Validate unsupported modules and JSON before creating any output files.
    saved_config = _save_config(model, tokenizer, config)
    config_json = json.dumps(saved_config, indent=2) + "\n"
    generation = deepcopy(getattr(model, "_checkpoint_generation_config", None))
    generation_json = (
        json.dumps(generation, indent=2) + "\n" if generation is not None else None
    )
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (path / "config.json").write_text(config_json)
    if generation_json is not None:
        (path / "generation_config.json").write_text(generation_json)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
