from dataclasses import asdict
from itertools import product

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten

from ttt_mlx import Model, ModelArgs, TTTLayer


def args_for(kind, **kwargs):
    return ModelArgs(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=43,
        mini_batch_size=4,
        ttt_layer_type=kind,
        **kwargs,
    )


def to_mlx(t):
    return mx.array(t.detach().float().cpu().numpy())


def compare(a, b, atol=2e-4, rtol=2e-4):
    if isinstance(b, torch.Tensor):
        b = b.detach().float().cpu().numpy()
    np.testing.assert_allclose(np.array(a), b, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "kind,share,gate,preconv",
    list(product(["linear", "mlp"], [False, True], [False, True], [False, True])),
)
def test_model_prefill_decode(reference, kind, share, gate, preconv):
    torch.manual_seed(14)
    args = args_for(kind, share_qk=share, use_gate=gate, pre_conv=preconv)
    ref = reference.TTTForCausalLM(reference.TTTConfig(**asdict(args))).eval()
    model = Model(args)
    model.load_weights(
        list(
            model.sanitize({k: to_mlx(v) for k, v in ref.state_dict().items()}).items()
        )
    )
    ids = torch.randint(0, args.vocab_size, (2, 13))
    with torch.no_grad():
        expected = ref(ids, use_cache=False).logits
        compare(model(to_mlx(ids).astype(mx.int32)), expected)
        # An unfinished initial mini-batch followed by single-token decode.
        cache = model.make_cache()
        ref_cache = None
        for begin, end in [(0, 5)] + [(i, i + 1) for i in range(5, 13)]:
            out = ref(ids[:, begin:end], cache_params=ref_cache, use_cache=True)
            ref_cache = out.cache_params
            actual = model(to_mlx(ids[:, begin:end]).astype(mx.int32), cache=cache)
            compare(actual, out.logits)
            for layer_index, c in enumerate(cache):
                for name in model.layers[
                    layer_index
                ].seq_modeling_block.fast_weight_names:
                    compare(
                        c.state[name],
                        ref_cache.ttt_params_dict[name + "_states"][layer_index],
                        atol=4e-4,
                    )
                    compare(
                        c.state[name + "_grad"],
                        ref_cache.ttt_params_dict[name + "_grad"][layer_index],
                        atol=6e-4,
                    )


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("compiled", [False, True])
def test_outer_gradients(reference, kind, compiled):
    torch.manual_seed(25)
    args = args_for(kind, compile_inner=compiled, use_gate=True, share_qk=True)
    cls = reference.TTTLinear if kind == "linear" else reference.TTTMLP
    ref = cls(reference.TTTConfig(**asdict(args)), layer_idx=0)
    layer = TTTLayer(args)
    weights = {k: to_mlx(v) for k, v in ref.state_dict().items()}
    for k in ("conv_q.weight", "conv_k.weight"):
        weights[k] = weights[k].transpose(0, 2, 1)
    layer.load_weights(list(weights.items()))
    # A nonzero learned token scale exercises clipping and update coefficients.
    values = torch.tensor([0.01, -0.02, 0.03, 0.02])
    with torch.no_grad():
        ref.learnable_token_idx.copy_(values)
    layer.learnable_token_idx = to_mlx(values)
    x = torch.randn(2, 9, args.hidden_size, requires_grad=True)
    target = torch.randn(2, 9, args.hidden_size)
    y = ref(x, position_ids=torch.arange(9)[None])
    loss = ((y - target) ** 2).mean()
    loss.backward()
    xm, tm = to_mlx(x), to_mlx(target)
    mloss, grads = nn.value_and_grad(
        layer, lambda model: mx.mean((model(xm) - tm) ** 2)
    )(layer)
    compare(mloss, loss, atol=2e-5)
    compare(
        mx.grad(lambda value: mx.mean((layer(value) - tm) ** 2))(xm),
        x.grad,
        atol=4e-4,
        rtol=3e-3,
    )
    grads = dict(tree_flatten(grads))
    for name, param in ref.named_parameters():
        grad = grads[name]
        if name in ("conv_q.weight", "conv_k.weight"):
            grad = grad.transpose(0, 2, 1)
        compare(grad, param.grad, atol=4e-4, rtol=3e-3)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_model_training_gradients(reference, kind):
    torch.manual_seed(55)
    args = args_for(kind, pre_conv=True, tie_word_embeddings=False)
    ref = reference.TTTForCausalLM(reference.TTTConfig(**asdict(args)))
    model = Model(args)
    model.load_weights(
        list(
            model.sanitize({k: to_mlx(v) for k, v in ref.state_dict().items()}).items()
        )
    )
    ids = torch.randint(0, args.vocab_size, (2, 9))
    logits = ref(ids[:, :-1], use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, args.vocab_size), ids[:, 1:].reshape(-1)
    )
    loss.backward()
    inputs = to_mlx(ids).astype(mx.int32)
    actual, grads = nn.value_and_grad(
        model,
        lambda m: nn.losses.cross_entropy(
            m(inputs[:, :-1]), inputs[:, 1:], reduction="mean"
        ),
    )(model)
    compare(actual, loss)
    grads = dict(tree_flatten(grads))
    for name, param in ref.named_parameters():
        grad = grads[name]
        if name.endswith(".conv.weight"):
            grad = grad.transpose(0, 2, 1)
        compare(grad, param.grad, atol=8e-4, rtol=5e-3)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_reference_head_dim_64(reference, kind):
    torch.manual_seed(103)
    args = ModelArgs(
        hidden_size=256,
        intermediate_size=384,
        num_hidden_layers=1,
        num_attention_heads=4,
        vocab_size=43,
        mini_batch_size=16,
        ttt_layer_type=kind,
    )
    ref = reference.TTTForCausalLM(reference.TTTConfig(**asdict(args))).eval()
    model = Model(args)
    model.load_weights(
        list(
            model.sanitize({k: to_mlx(v) for k, v in ref.state_dict().items()}).items()
        )
    )
    ids = torch.randint(0, 43, (1, 67))
    with torch.no_grad():
        expected = ref(ids, use_cache=False).logits
        compare(model(mx.array(ids.numpy())), expected)
        # Reference supports aligned prefill and single-token continuation.
        out = ref(ids[:, :64], use_cache=True)
        cache = model.make_cache()
        compare(model(mx.array(ids[:, :64].numpy()), cache=cache), out.logits)
        ref_cache = out.cache_params
        for i in range(64, 67):
            out = ref(ids[:, i : i + 1], cache_params=ref_cache, use_cache=True)
            compare(model(mx.array(ids[:, i : i + 1].numpy()), cache=cache), out.logits)
