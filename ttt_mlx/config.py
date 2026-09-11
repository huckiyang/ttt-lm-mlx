from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "ttt"
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    ttt_layer_type: str = "linear"
    ttt_base_lr: float = 1.0
    mini_batch_size: int = 16
    share_qk: bool = False
    use_gate: bool = False
    pre_conv: bool = False
    conv_kernel: int = 4
    # Keep recurrent weights and accumulators in FP32 with low precision projections.
    state_dtype: str = "float32"
    compile_inner: bool = True

    def __post_init__(self):
        if self.model_type != "ttt":
            raise ValueError("model_type must be ttt")
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "mini_batch_size",
            "conv_kernel",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.ttt_layer_type not in ("linear", "mlp"):
            raise ValueError("ttt_layer_type must be linear or mlp")
        if self.hidden_act != "silu":
            raise ValueError("The backbone currently supports hidden_act=silu")
        if self.state_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Invalid state_dtype")
        if self.rope_theta <= 0 or self.ttt_base_lr < 0 or self.rms_norm_eps <= 0:
            raise ValueError(
                "Invalid RoPE base, learning rate, or normalization epsilon"
            )

    @classmethod
    def from_dict(cls, params):
        if params.get("rope_scaling") is not None:
            raise ValueError(
                "Scaled RoPE is not part of the reference TTT architecture"
            )
        return super().from_dict(params)
