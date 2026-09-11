import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from ttt_mlx import (
    Model,
    ModelArgs,
    load_cache,
    load_model,
    save_cache,
    save_model,
    prefill,
)


def small_model(kind="linear", **kwargs):
    mx.random.seed(17)
    return Model(
        ModelArgs(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=43,
            mini_batch_size=4,
            ttt_layer_type=kind,
            **kwargs,
        )
    )


def close(x, y, atol=3e-4):
    np.testing.assert_allclose(np.array(x), np.array(y), atol=atol, rtol=3e-4)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("kernel", [1, 4])
def test_arbitrary_chunks_cache_and_causality(kind, kernel, tmp_path):
    model = small_model(
        kind, share_qk=True, pre_conv=True, use_gate=True, conv_kernel=kernel
    )
    ids = mx.random.randint(0, 43, (2, 19))
    expected = model(ids)
    caches = model.make_cache()
    pieces = []
    start = 0
    initial_weights = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    for length in [3, 6, 1, 7, 2]:
        pieces.append(model(ids[:, start : start + length], cache=caches))
        mx.eval(pieces[-1], [c.state for c in caches])
        start += length
        assert all(c.offset == start for c in caches)
    close(mx.concatenate(pieces, axis=1), expected)
    close(model(ids[:, :7]), expected[:, :7])
    for name, value in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(np.array(value), initial_weights[name])
    path = tmp_path / "cache.safetensors"
    save_cache(path, caches)
    restored = load_cache(path)
    clone = [c.clone() for c in caches]
    next_ids = mx.array([[2, 3, 4], [3, 4, 5]])
    next_logits = model(next_ids, cache=restored)
    close(model(next_ids, cache=clone), next_logits)
    assert all(c.offset == 19 for c in caches)
    assert not any(c.is_trimmable() for c in caches)
    nbytes = sum(c.nbytes for c in clone)
    model(mx.ones((2, 25), dtype=mx.int32), cache=clone)
    mx.eval([c.state for c in clone])
    assert sum(c.nbytes for c in clone) == nbytes
    for c in caches:
        c.reset()
        assert c.empty() and c.nbytes == 0
    close(model(ids, cache=caches), expected)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_mlx_lm_generation_and_checkpoint(kind, tmp_path):
    model = small_model(kind)
    model.eval()
    ids = mx.array([1, 2, 3, 4, 5, 6, 7])
    cache = make_prompt_cache(model)
    actual = [
        token
        for token, _ in generate_step(
            ids, model, max_tokens=5, prompt_cache=cache, prefill_step_size=3
        )
    ]
    full = ids[None]
    expected = []
    for _ in range(5):
        token = int(mx.argmax(model(full)[:, -1]).item())
        expected.append(token)
        full = mx.concatenate([full, mx.array([[token]])], axis=1)
    assert actual == expected
    save_model(tmp_path, model)
    restored, _ = load_model(tmp_path)
    close(restored(ids[None]), model(ids[None]))
    with pytest.raises(FileExistsError):
        save_model(tmp_path, model)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_mixed_precision(kind, dtype):
    model = small_model(kind)
    model.set_dtype(dtype)
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    expected = model(ids)
    cache = model.make_cache()
    actual = mx.concatenate(
        [model(ids[:, :3], cache=cache), model(ids[:, 3:], cache=cache)], axis=1
    )
    assert bool(mx.all(mx.isfinite(actual)))
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=0.02,
        rtol=0.02,
    )
    assert all(c.state["W1"].dtype == mx.float32 for c in cache)


def test_invalid_cache_is_rejected_before_mutation():
    model = small_model(pre_conv=True)
    cache = model.make_cache()
    model(mx.array([[1, 2]]), cache=cache)
    with pytest.raises(ValueError, match="batch size"):
        model(mx.array([[1], [2]]), cache=cache)
    assert all(c.offset == 2 for c in cache)
    with pytest.raises(ValueError, match="one TTTCache"):
        model(mx.array([[1]]), cache=cache[:1])


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_training_step(kind):
    import mlx.optimizers as optim

    model = small_model(kind)
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

    def loss_fn(m):
        return nn.losses.cross_entropy(m(ids[:, :-1]), ids[:, 1:], reduction="mean")

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer = optim.Adam(learning_rate=1e-4)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    assert float(loss_fn(model)) < float(loss)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_gradients_across_streaming_calls(kind):
    model = small_model(kind, share_qk=True, pre_conv=True)
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

    def streamed(m):
        cache = m.make_cache()
        first = m(ids[:, :3], cache=cache)
        rest = m(ids[:, 3:], cache=cache)
        return mx.mean(mx.concatenate([first, rest], axis=1) ** 2)

    loss, grads = nn.value_and_grad(model, lambda m: mx.mean(m(ids) ** 2))(model)
    streamed_loss, streamed_grads = nn.value_and_grad(model, streamed)(model)
    close(loss, streamed_loss)
    expected = dict(tree_flatten(grads))
    for name, value in tree_flatten(streamed_grads):
        close(value, expected[name], atol=8e-4)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_prefill_realistic_head_size(kind):
    mx.random.seed(41)
    model = Model(
        ModelArgs(
            hidden_size=256,
            intermediate_size=384,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=43,
            mini_batch_size=16,
            ttt_layer_type=kind,
        )
    )
    ids = mx.random.randint(0, 43, (1, 67))
    full = model(ids)
    mx.eval(full)
    logits, cache = prefill(model, ids, chunk_size=21)
    close(logits, full[:, -1:])
    for c in cache:
        c.detach()
    extended = mx.concatenate([ids, mx.array([[2, 3]])], axis=1)
    close(model(extended)[:, -2:], model(mx.array([[2, 3]]), cache=cache))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(hidden_size=31),
        dict(num_attention_heads=0),
        dict(mini_batch_size=0),
        dict(ttt_layer_type="attention"),
        dict(state_dtype="int8"),
        dict(conv_kernel=0),
        dict(hidden_act="relu"),
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        ModelArgs(**kwargs)
