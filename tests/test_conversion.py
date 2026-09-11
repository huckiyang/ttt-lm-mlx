import json
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
import torch

from ttt_mlx import ModelArgs, load, load_model
from ttt_mlx.convert import convert


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("format", ["safetensors", "bin"])
def test_reference_checkpoint_conversion(reference, kind, format, tmp_path):
    args = ModelArgs(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=43,
        mini_batch_size=4,
        ttt_layer_type=kind,
        share_qk=True,
        pre_conv=True,
        use_gate=True,
    )
    torch.manual_seed(81)
    ref = reference.TTTForCausalLM(reference.TTTConfig(**asdict(args))).eval()
    source, dest = tmp_path / "source", tmp_path / "mlx"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(args) | {"eos_token_id": 2}))
    weights = list(ref.state_dict().items())
    for i, part in enumerate(
        [weights[: len(weights) // 2], weights[len(weights) // 2 :]]
    ):
        if format == "safetensors":
            mx.save_safetensors(
                str(source / f"model-{i}.safetensors"),
                {k: mx.array(v.detach().numpy()) for k, v in part},
            )
        else:
            torch.save(dict(part), source / f"pytorch_model-{i}.bin")
    # A local tokenizer makes the full model/tokenizer loader test network-free.
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "hello": 1, "[EOS]": 2}, unk_token="[UNK]")
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(source)
    convert(source, dest)
    model, loaded_tokenizer = load(dest)
    assert loaded_tokenizer.encode("hello", add_special_tokens=False) == [1]
    assert json.loads((dest / "config.json").read_text())["eos_token_id"] == 2
    ids = torch.randint(0, 43, (1, 11))
    with torch.no_grad():
        expected = ref(ids).logits.numpy()
    actual = model(mx.array(ids.numpy()))
    np.testing.assert_allclose(np.array(actual), expected, atol=3e-4, rtol=3e-4)
    half_dest = tmp_path / "half"
    convert(source, half_dest, dtype="float16")
    half, _ = load_model(half_dest)
    assert half.model.embed_tokens.weight.dtype == mx.float16
    assert half.layers[0].seq_modeling_block.W1.dtype == mx.float32
    half_output = half(mx.array(ids.numpy()))
    assert bool(mx.all(mx.isfinite(half_output)))
    np.testing.assert_allclose(
        np.array(half_output.astype(mx.float32)), expected, atol=0.02, rtol=0.02
    )
