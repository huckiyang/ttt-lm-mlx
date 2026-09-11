import mlx.core as mx
import mlx.nn as nn

from .cache import TTTCache
from .config import ModelArgs
from .functional import (
    compiled_linear_chunk,
    compiled_mlp_chunk,
    gelu,
    linear_chunk,
    mlp_chunk,
)


def causal_conv(conv, x, cache, key):
    k = conv.weight.shape[1]
    if k == 1:
        return conv(x)
    history = cache.state.get(key) if cache is not None else None
    if history is None:
        history = mx.zeros((x.shape[0], k - 1, x.shape[-1]), dtype=x.dtype)
    full = mx.concatenate([history, x], axis=1)
    if cache is not None:
        # Gather into compact storage; a slice can retain the entire prefill buffer.
        indices = mx.arange(full.shape[1] - (k - 1), full.shape[1])
        cache.state[key] = mx.take(full, indices, axis=1)
    return conv(full)


class TTTLayer(nn.Module):
    """TTT sequence mixer with differentiable, request-local fast weights.

    With no cache each call starts a fresh sequence. A cache continues it and
    can retain an autodiff graph; call cache.detach() for truncated training.
    Inputs are unpadded [batch, sequence, hidden_size] arrays.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        d, h = args.hidden_size // args.num_attention_heads, args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        if args.share_qk:
            self.conv_q = nn.Conv1d(
                args.hidden_size,
                args.hidden_size,
                args.conv_kernel,
                groups=args.hidden_size,
            )
            self.conv_k = nn.Conv1d(
                args.hidden_size,
                args.hidden_size,
                args.conv_kernel,
                groups=args.hidden_size,
            )
        else:
            self.k_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        if args.use_gate:
            self.g_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.learnable_token_idx = mx.zeros((args.mini_batch_size,))
        self.learnable_ttt_lr_weight = mx.random.normal((h, 1, args.hidden_size)) * 0.02
        self.learnable_ttt_lr_bias = mx.zeros((h, 1))
        self.ttt_norm_weight = mx.ones((h, d))
        self.ttt_norm_bias = mx.zeros((h, d))
        self.post_norm = nn.LayerNorm(args.hidden_size, eps=1e-6)
        inner = d if args.ttt_layer_type == "linear" else 4 * d
        self.W1 = mx.random.normal((h, d, inner)) * 0.02
        self.b1 = mx.zeros((h, 1, inner))
        if args.ttt_layer_type == "mlp":
            self.W2 = mx.random.normal((h, inner, d)) * 0.02
            self.b2 = mx.zeros((h, 1, d))

    @property
    def fast_weight_names(self):
        return (
            ("W1", "b1")
            if self.args.ttt_layer_type == "linear"
            else ("W1", "b1", "W2", "b2")
        )

    def _rope(self, x, offset):
        d = x.shape[-1]
        pos = mx.arange(offset, offset + x.shape[-2]) % self.args.mini_batch_size
        freq = self.args.rope_theta ** (-mx.arange(0, d, 2, dtype=mx.float32) / d)
        angles = pos[:, None] * freq[None, :]
        cos, sin = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return mx.stack(
            [even * cos - odd * sin, even * sin + odd * cos], axis=-1
        ).reshape(x.shape)

    def __call__(self, x, cache=None):
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[-1] != self.args.hidden_size:
            raise ValueError("Expected nonempty [batch, sequence, hidden_size] inputs")
        if cache is not None and not isinstance(cache, TTTCache):
            raise TypeError("Expected TTTCache")
        b, length, width = x.shape
        h, d = self.args.num_attention_heads, width // self.args.num_attention_heads
        offset = cache.offset if cache is not None else 0
        names = self.fast_weight_names
        dtype = getattr(mx, self.args.state_dtype)
        if cache is not None and "W1" in cache.state:
            if cache.state["W1"].shape[0] != b:
                raise ValueError("Cache batch size differs from input batch size")
            params = tuple(cache.state[n] for n in names)
            accum = tuple(cache.state[n + "_grad"] for n in names)
        else:
            params = tuple(
                mx.broadcast_to(
                    getattr(self, n).astype(dtype)[None], (b, *getattr(self, n).shape)
                )
                for n in names
            )
            accum = tuple(mx.zeros_like(p) for p in params)

        q, v = self.q_proj(x), self.v_proj(x)
        if self.args.share_qk:
            q, k = (
                causal_conv(self.conv_q, q, cache, "conv_q"),
                causal_conv(self.conv_k, q, cache, "conv_k"),
            )
        else:
            k = self.k_proj(x)
        q, k, v = (a.reshape(b, length, h, d).transpose(0, 2, 1, 3) for a in (q, k, v))
        q, k = self._rope(q, offset).astype(dtype), self._rope(k, offset).astype(dtype)
        v = v.astype(dtype)
        lr = mx.sigmoid(
            x.astype(dtype) @ self.learnable_ttt_lr_weight[:, 0].astype(dtype).T
            + self.learnable_ttt_lr_bias[:, 0].astype(dtype)
        )
        lr = lr.transpose(0, 2, 1)[..., None] * (self.args.ttt_base_lr / d)
        norm_w, norm_b = (
            self.ttt_norm_weight[:, None].astype(dtype),
            self.ttt_norm_bias[:, None].astype(dtype),
        )
        if self.args.ttt_layer_type == "linear":
            step = compiled_linear_chunk if self.args.compile_inner else linear_chunk
        else:
            step = compiled_mlp_chunk if self.args.compile_inner else mlp_chunk
        outputs = []
        start, mb = 0, self.args.mini_batch_size
        while start < length:
            position = (offset + start) % mb
            count = min(mb - position, length - start)
            end = start + count
            coefficient = 1 / mx.arange(
                position + 1, position + count + 1, dtype=dtype
            ) + self.learnable_token_idx[position : position + count].astype(dtype)
            # torch.clamp_min has derivative one at zero; mx.maximum has zero.
            token = mx.where(coefficient >= 0, coefficient, 0)
            token = token[None, None, :, None]
            y, accum = step(
                params,
                accum,
                q[:, :, start:end],
                k[:, :, start:end],
                v[:, :, start:end],
                lr[:, :, start:end],
                token,
                norm_w,
                norm_b,
            )
            outputs.append(y)
            if position + count == mb:
                params = tuple(
                    p - token[..., -1:, :] * g for p, g in zip(params, accum)
                )
                accum = tuple(mx.zeros_like(p) for p in params)
            start = end
        if cache is not None:
            cache.state.update(zip(names, params))
            cache.state.update((n + "_grad", g) for n, g in zip(names, accum))
            cache.offset += length
        y = (
            mx.concatenate(outputs, axis=2)
            .transpose(0, 2, 1, 3)
            .reshape(b, length, width)
        )
        y = self.post_norm(y.astype(x.dtype))
        if self.args.use_gate:
            y = y * gelu(self.g_proj(x))
        return self.o_proj(y)


class TTTLinear(TTTLayer):
    def __init__(self, args):
        if args.ttt_layer_type != "linear":
            raise ValueError("TTTLinear requires ttt_layer_type=linear")
        super().__init__(args)


class TTTMLP(TTTLayer):
    def __init__(self, args):
        if args.ttt_layer_type != "mlp":
            raise ValueError("TTTMLP requires ttt_layer_type=mlp")
        super().__init__(args)
