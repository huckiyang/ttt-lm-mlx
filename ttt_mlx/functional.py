"""Pure, differentiable inner updates. Arrays use [batch, head, token, feature]."""

import mlx.core as mx


def layer_norm(x, weight, bias, eps=1e-6):
    centered = x - mx.mean(x, axis=-1, keepdims=True)
    normalized = centered * mx.rsqrt(
        mx.mean(centered * centered, axis=-1, keepdims=True) + eps
    )
    return normalized * weight + bias


def ln_l2_grad(x, target, weight, bias, eps=1e-6):
    centered = x - mx.mean(x, axis=-1, keepdims=True)
    inv_std = mx.rsqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + eps)
    normalized = centered * inv_std
    grad = (normalized * weight + bias - target) * weight
    return (
        grad
        - mx.mean(grad, axis=-1, keepdims=True)
        - normalized * mx.mean(grad * normalized, axis=-1, keepdims=True)
    ) * inv_std


def gelu(x):
    return 0.5 * x * (1 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def gelu_grad(x):
    t = mx.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (1 - t * t) * (0.79788456 + 0.1070322243 * x * x) + 0.5 * (1 + t)


def _read_and_accumulate(q, k, grad, w, b, gw, gb, lr, token):
    # Include gradients from an earlier call in this same mini-batch.
    eta = mx.tril(token * mx.swapaxes(lr, -1, -2))
    z = q @ w + b - token * (q @ gw + gb)
    z = z - (eta * (q @ mx.swapaxes(k, -1, -2))) @ grad - eta @ grad
    gw = gw + mx.swapaxes(k * lr, -1, -2) @ grad
    gb = gb + mx.sum(lr * grad, axis=-2, keepdims=True)
    return z, gw, gb


def linear_chunk(params, accum, q, k, v, lr, token, norm_weight, norm_bias):
    w, b = params
    grad = ln_l2_grad(k @ w + b, v - k, norm_weight, norm_bias)
    z, gw, gb = _read_and_accumulate(q, k, grad, w, b, *accum, lr, token)
    return q + layer_norm(z, norm_weight, norm_bias), (gw, gb)


def mlp_chunk(params, accum, q, k, v, lr, token, norm_weight, norm_bias):
    w1, b1, w2, b2 = params
    gw1, gb1, gw2, gb2 = accum
    z1 = k @ w1 + b1
    x2 = gelu(z1)
    grad2 = ln_l2_grad(x2 @ w2 + b2, v - k, norm_weight, norm_bias)
    grad1 = (grad2 @ mx.swapaxes(w2, -1, -2)) * gelu_grad(z1)
    z1_bar, gw1, gb1 = _read_and_accumulate(q, k, grad1, w1, b1, gw1, gb1, lr, token)
    z2_bar, gw2, gb2 = _read_and_accumulate(
        gelu(z1_bar), x2, grad2, w2, b2, gw2, gb2, lr, token
    )
    return q + layer_norm(z2_bar, norm_weight, norm_bias), (gw1, gb1, gw2, gb2)


compiled_linear_chunk = mx.compile(linear_chunk)
compiled_mlp_chunk = mx.compile(mlp_chunk)
