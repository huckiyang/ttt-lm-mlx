"""Regressions for retained convolution storage, checkpoint metadata, and clipping."""

import gc
import json
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from mlx_lm.utils import dequantize_model
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from ttt_mlx import Model, ModelArgs, TTTCache, TTTLayer, load, load_model, save_model
from ttt_mlx.layer import causal_conv


def small_model(kind="linear", tied=True):
    return Model(
        ModelArgs(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=32,
            mini_batch_size=4,
            ttt_layer_type=kind,
            tie_word_embeddings=tied,
        )
    )


def local_tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "hello": 1, "[EOS]": 2, "[END]": 3}, unk_token="[UNK]"
            )
        ),
        unk_token="[UNK]",
        eos_token="[EOS]",
    )


@pytest.mark.parametrize("batch", [1, 2])
def test_convolution_history_owns_compact_storage(batch):
    conv = nn.Conv1d(64, 64, 4, groups=64)
    mx.eval(conv.parameters())
    retained_sizes = []
    for length in (32, 8192):
        gc.collect()
        mx.synchronize()
        baseline = mx.get_active_memory()
        cache = TTTCache()
        x = mx.random.normal((batch, length, 64))
        expected = np.array(x[:, -3:])
        y = causal_conv(conv, x, cache, "history")
        mx.eval(y, cache.state)
        snapshot = cache.clone()
        del x, y, cache
        gc.collect()
        mx.synchronize()
        retained = mx.get_active_memory() - baseline
        logical = snapshot.nbytes
        np.testing.assert_array_equal(np.array(snapshot.state["history"]), expected)
        assert logical == batch * 3 * 64 * 4
        # Permit allocator granularity, but not storage proportional to the prompt.
        assert retained <= logical + 4096, (length, retained, logical)
        retained_sizes.append(retained)
        del snapshot
    assert abs(retained_sizes[1] - retained_sizes[0]) <= 4096


@pytest.mark.parametrize("with_tokenizer", [False, True])
def test_loaded_metadata_survives_save(tmp_path, with_tokenizer):
    source = tmp_path / "source"
    tokenizer = local_tokenizer()
    metadata = {
        "eos_token_id": [2, 3],
        "bos_token_id": 1,
        "pad_token_id": 0,
        "max_position_embeddings": 8192,
        "task_specific_params": {"name": "test"},
    }
    save_model(source, small_model(), tokenizer, config=metadata)
    generation = {"eos_token_id": [2, 3], "temperature": 0.7, "top_p": 0.9}
    (source / "generation_config.json").write_text(json.dumps(generation))
    loaded, wrapped = load(source)
    assert wrapped.eos_token_ids == {2, 3}
    assert "_checkpoint_config" not in loaded.state
    # Saving must preserve metadata even without a tokenizer argument.
    for iteration in range(2):
        destination = tmp_path / f"roundtrip-{iteration}"
        save_model(destination, loaded, wrapped if with_tokenizer else None)
        _, config = load_model(destination)
        for key, value in metadata.items():
            assert config[key] == value
        assert (
            json.loads((destination / "generation_config.json").read_text())
            == generation
        )
        loaded, _ = load_model(destination)
        if with_tokenizer:
            loaded, wrapped = load(destination)
            assert wrapped.eos_token_ids == {2, 3}


def test_runtime_eos_updates_survive_save(tmp_path):
    source = tmp_path / "source"
    save_model(source, small_model(), local_tokenizer())
    loaded, tokenizer = load(source)
    tokenizer.eos_token_ids = {2, 3}
    save_model(tmp_path / "updated", loaded, tokenizer)
    _, restored = load(tmp_path / "updated")
    assert restored.eos_token_ids == {2, 3}


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_quantized_checkpoint_roundtrip(tmp_path, kind, tied, mixed):
    mx.random.seed(303)
    model = small_model(kind, tied)
    predicate = None
    if mixed:

        def predicate(name, module):
            if not hasattr(module, "to_quantized") or name.endswith("k_proj"):
                return False
            return {
                "bits": 8 if name.endswith("v_proj") else 4,
                "group_size": 32,
                "mode": "affine",
            }

    nn.quantize(model, group_size=32, bits=4, class_predicate=predicate)
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    expected = model(ids)
    mx.eval(expected)
    original_weights = {
        name: np.array(value) for name, value in tree_flatten(model.parameters())
    }
    for iteration in range(2):
        path = tmp_path / str(iteration)
        save_model(path, model)
        model, config = load_model(path)
        assert "quantization" in config
        for name, value in tree_flatten(model.parameters()):
            np.testing.assert_array_equal(np.array(value), original_weights[name])
        np.testing.assert_allclose(
            np.array(model(ids)), np.array(expected), atol=2e-5, rtol=2e-5
        )
        if mixed:
            ttt = model.layers[0].seq_modeling_block
            assert ttt.v_proj.bits == 8
            assert isinstance(ttt.k_proj, nn.Linear)


def test_dequantized_save_removes_old_quantization(tmp_path):
    model = small_model()
    nn.quantize(model, group_size=32, bits=4)
    save_model(tmp_path / "quantized", model)
    model, _ = load_model(tmp_path / "quantized")
    model = dequantize_model(model)
    expected = model(mx.array([[1, 2, 3]]))
    save_model(tmp_path / "float", model)
    reloaded, config = load_model(tmp_path / "float")
    assert (
        not {"quantization", "quantization_config", "quantize_activations"}
        & config.keys()
    )
    np.testing.assert_allclose(
        np.array(reloaded(mx.array([[1, 2, 3]]))), np.array(expected), atol=2e-5
    )


def test_unsupported_quantization_fails_before_writing(tmp_path):
    model = small_model()
    layer = model.layers[0].seq_modeling_block
    layer.q_proj = nn.QQLinear.from_linear(layer.q_proj, mode="nvfp4")
    destination = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="QQLinear"):
        save_model(destination, model)
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("commit_at_zero", [False, True])
def test_clipped_token_coefficient_gradient_parity(
    reference, kind, compiled, commit_at_zero
):
    torch.manual_seed(200)
    args = ModelArgs(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=32,
        mini_batch_size=4,
        ttt_layer_type=kind,
        compile_inner=compiled,
    )
    cls = reference.TTTLinear if kind == "linear" else reference.TTTMLP
    ref = cls(reference.TTTConfig(**asdict(args)), layer_idx=0)
    # Exercise negative, exactly zero, and positive effective scales. The
    # zero commit scale also tests gradients into the next mini-batch.
    shifts = [-1.0, -0.7, 0.02, -0.25] if commit_at_zero else [-1.0, -0.7, -1 / 3, 0.2]
    with torch.no_grad():
        ref.learnable_token_idx.copy_(torch.tensor(shifts))
    layer = TTTLayer(args)
    layer.load_weights(
        [
            (name, mx.array(value.detach().numpy()))
            for name, value in ref.state_dict().items()
        ]
    )
    x = torch.randn(1, 9, 32, requires_grad=True)
    target = torch.randn_like(x)
    expected = ref(x, position_ids=torch.arange(9)[None])
    loss = ((expected - target) ** 2).mean()
    loss.backward()
    xm, tm = mx.array(x.detach().numpy()), mx.array(target.numpy())
    actual, grads = nn.value_and_grad(layer, lambda m: mx.mean((m(xm) - tm) ** 2))(
        layer
    )
    np.testing.assert_allclose(
        np.array(layer(xm)), expected.detach().numpy(), atol=2e-4, rtol=2e-4
    )
    np.testing.assert_allclose(np.array(actual), loss.detach().numpy(), atol=2e-5)
    input_grad = mx.grad(lambda value: mx.mean((layer(value) - tm) ** 2))(xm)
    np.testing.assert_allclose(
        np.array(input_grad), x.grad.numpy(), atol=4e-4, rtol=3e-3
    )
    grads = dict(tree_flatten(grads))
    assert abs(float(ref.learnable_token_idx.grad[0])) > 0.01
    for name, param in ref.named_parameters():
        np.testing.assert_allclose(
            np.array(grads[name]),
            param.grad.numpy(),
            atol=4e-4,
            rtol=3e-3,
            err_msg=name,
        )
