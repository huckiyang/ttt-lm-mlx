import mlx.core as mx
import mlx.nn as nn

from .cache import TTTCache
from .config import ModelArgs
from .layer import TTTLayer, causal_conv


class SwiGluMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Conv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.conv = nn.Conv1d(
            args.hidden_size,
            args.hidden_size,
            args.conv_kernel,
            groups=args.hidden_size,
        )

    def __call__(self, x, cache):
        return causal_conv(self.conv, self.norm(x), cache, "pre_conv")


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_modeling_block = TTTLayer(args)
        self.mlp = SwiGluMLP(args)
        self.seq_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.pre_conv:
            self.conv = Conv(args)

    def __call__(self, x, cache):
        if "conv" in self:
            x = x + self.conv(x, cache)
        x = x + self.seq_modeling_block(self.seq_norm(x), cache)
        return x + self.mlp(self.ffn_norm(x))


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Block(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache, input_embeddings=None):
        x = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        for layer, c in zip(self.layers, cache):
            x = layer(x, c)
        return self.norm(x)


class Model(nn.Module):
    """MLX-LM causal language model matching the reference TTT backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Backbone(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight = (
                    mx.random.normal(module.weight.shape) * args.initializer_range
                )

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if inputs.ndim != 2 or inputs.shape[1] == 0:
            raise ValueError("Expected nonempty [batch, sequence] token IDs")
        if cache is None:
            cache = [None] * len(self.layers)
        else:
            if len(cache) != len(self.layers) or any(
                not isinstance(c, TTTCache) for c in cache
            ):
                raise ValueError("Expected one TTTCache per layer")
            if len({c.offset for c in cache}) != 1:
                raise ValueError("Layer cache offsets must match")
            for c in cache:
                if "W1" in c.state and c.state["W1"].shape[0] != inputs.shape[0]:
                    raise ValueError("Cache batch size differs from input batch size")
        if input_embeddings is not None and input_embeddings.shape != (
            *inputs.shape,
            self.args.hidden_size,
        ):
            raise ValueError("input_embeddings shape does not match inputs")
        x = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(x)
        return self.lm_head(x)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [TTTCache() for _ in self.layers]

    def sanitize(self, weights):
        weights = dict(weights)
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for name, value in list(weights.items()):
            if name.endswith((".rotary_emb.inv_freq", ".token_idx")):
                weights.pop(name)
            elif (
                name.endswith((".conv.weight", ".conv_q.weight", ".conv_k.weight"))
                and value.shape[-1] != 1
            ):
                weights[name] = value.transpose(0, 2, 1)
        return weights
