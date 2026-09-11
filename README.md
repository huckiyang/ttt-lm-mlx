# TTT-LM for MLX

Native MLX **TTT-Linear and TTT-MLP** sequence layers, plus a causal language
model that uses MLX-LM's generation and checkpoint-loading interfaces. The inner
learner adapts request-local fast weights from the sequence; outer training
differentiates through those updates to learn the model's parameters.

This implements the architecture in the supplied
[PyTorch reference](https://github.com/test-time-training/ttt-lm-pytorch),
with the [JAX implementation](https://github.com/test-time-training/ttt-lm-jax)
and [inference kernels](https://github.com/test-time-training/ttt-lm-kernels)
as architectural references. PyTorch is used in tests and optional `.bin`
checkpoint conversion. The runtime forward and backward paths use MLX.

Implemented features:

- Linear and two-layer GELU inner learners, learned per-head learning rates,
  learned token coefficients, inner LayerNorm, and mini-batch-local RoPE.
- Causal dual-form mini-batch computation, with optional `mx.compile` fusion.
- Arbitrary prefill/decode chunk lengths, including calls that cross an
  unfinished mini-batch; constant-size recurrent state after evaluation.
- Shared Q/K projections with separate causal convolutions, output gating,
  optional residual pre-convolution, RMSNorm, and the SwiGLU backbone.
- Full outer-training gradients, tied or untied output embeddings, FP32
  recurrent state by default, and FP16/BF16 projection weights.
- MLX-LM `generate_step`/`generate`/`stream_generate`, local checkpoint loading
  and conversion, and prompt-state snapshots and persistence.

## Setup

On an Apple silicon Mac with Metal available, from this directory:

```bash
python -m pip install -e '.[test]' -e ../../mlx-lm
python -m pytest tests -q
```

The second editable path selects the user's existing MLX-LM checkout. Omit it
to use the installed MLX-LM package. Tests automatically prefer `../../mlx-lm`
and `../ttt-lm-pytorch`; override with `MLX_LM_PATH` and `TTT_TORCH_PATH`.
The parity tests fail if the reference is missing rather than silently skipping.
Metal execution may need permission outside a restricted application sandbox.

## Layer and streaming model

```python
import mlx.core as mx
from ttt_mlx import Model, ModelArgs, TTTLayer, TTTCache, prefill

args = ModelArgs(
    hidden_size=256, intermediate_size=768, num_hidden_layers=2,
    num_attention_heads=4, vocab_size=256,
    mini_batch_size=16, ttt_layer_type="mlp",
)
layer = TTTLayer(args)
state = TTTCache()
y = layer(mx.random.normal((1, 23, 256)), cache=state)
mx.eval(y, state.state)

model = Model(args)
model.eval()
tokens = mx.array([[1, 2, 3, 4, 5]])
logits, cache = prefill(model, tokens, chunk_size=3)
next_logits = model(mx.array([[6]]), cache=cache)
mx.eval(next_logits, [c.state for c in cache])
```

These weights are random. Use a trained TTT checkpoint for meaningful text.
Calling without a cache starts a new sequence. With a cache, all rows must
belong to continuing, unpadded sequences of the same length and mini-batch
phase. Fast weights never overwrite model parameters.

For direct long-prompt inference, `prefill` evaluates each chunk and returns
only last-token logits plus the cache. Direct `model(tokens)` returns all logits
and retains the graph needed for training. `model.make_cache()` produces one
state object per layer; ordinary MLX-LM generation uses this method automatically.

```python
from ttt_mlx import save_cache, load_cache

save_cache("prompt.safetensors", cache)
restored = load_cache("prompt.safetensors")
branch = [c.clone() for c in restored]
for c in restored:
    c.reset()
```

Use this package's cache persistence helpers: MLX-LM's built-in cache loader
looks up cache types in its own module and does not register external classes.
Snapshots share immutable arrays until subsequent calls replace them.

## Load and generate

Unquantized reference `model*.safetensors` checkpoints can be loaded directly;
the model sanitizes PyTorch convolution layouts. To convert safetensors or
`pytorch_model*.bin` shards and copy local tokenizer assets:

```bash
python -m ttt_mlx.convert --source /path/to/torch-ttt --output /path/to/mlx-ttt
# Optional: --dtype float16 or --dtype bfloat16
```

Conversion retains FP32 learned fast-weight initializers and update-rate/norm
parameters. It uses strict weight matching and refuses to overwrite an existing
checkpoint. No model or tokenizer downloads are implicit.

```python
from ttt_mlx import load
from mlx_lm import generate

model, tokenizer = load("/path/to/mlx-ttt")
text = generate(model, tokenizer, prompt="Hello", max_tokens=64)
print(text)
```

`ttt_mlx.load_model(path)` returns `(model, config)` without a tokenizer.
`ttt_mlx.save_model(path, model, tokenizer=None, config=None)` saves native weights
and retains loaded checkpoint metadata, including multiple EOS IDs and an existing
`generation_config.json`. The keyword-only `config` argument supplies metadata for
a new model. When an MLX-LM tokenizer wrapper is passed, its current EOS IDs are
saved in `config.json`; an existing `generation_config.json` is preserved separately.

Weight quantization is recorded from the actual `QuantizedLinear` and
`QuantizedEmbedding` modules, including partial quantization and per-module
settings. Both tied and untied models can round-trip through `nn.quantize`,
`save_model`, and `load_model`. Saving activation-quantized `QQLinear` modules
is currently rejected before writing any files.

Loading uses MLX-LM's `get_model_classes` hook; the upstream checkout is not modified.
The upstream `mlx_lm.load`/CLI does not discover this external model on its own.

## Train and verify parity

```bash
python examples/train.py
python -m pytest tests/test_parity.py -q
python -m pytest tests -q
```

The training example takes real optimizer steps on a small synthetic batch.
Replace the tokens with your corpus for training. The loss is next-token
cross entropy; the inner learning objective remains reconstruction of `V - K`.
The analytic inner gradient is built from differentiable MLX operations, so
outer gradients include the derivative through the inner learning step.

For full-sequence training, use no cache. For training across calls, keep the
cache inside the differentiated function; `cache.detach()` explicitly truncates
its gradient history. Evaluating a cache is an execution boundary, not a request
to stop gradients. The Python mini-batch loop is unrolled into the lazy graph;
training memory still grows with sequence length.

Tests load identical weights from the actual supplied PyTorch implementation
and compare outputs, recurrent weights/accumulators, layer/input gradients,
and full-model cross-entropy gradients. All combinations of sharing, gating,
and pre-convolution are covered. Further tests check arbitrary streaming splits,
causality, independent snapshots, save/restore, conversion, mixed precision,
training across calls, and MLX-LM generation. Regression tests measure actual
retained convolution memory, check checkpoint metadata and quantized round trips,
and compare gradients for negative, zero, and positive effective token coefficients.

FP32 forward tolerances start at `atol=rtol=2e-4`; outer gradients allow up to
`atol=8e-4, rtol=5e-3`. Low-precision smoke comparisons use `2e-2` and do not
establish pretrained-model quality parity. Under Transformers 5 the test fixture
adapts the reference class's tied-weight metadata format; its numerical code is
unchanged.

## MLX and unified memory

Fast weights and accumulators remain MLX arrays in Apple silicon's shared
memory pool. There are no host transfers in the update loop. Operations run on
the selected MLX device; sharing physical memory does not eliminate graph
synchronization or guarantee zero-copy interchange with PyTorch. See MLX's
[unified-memory documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).

The dual form avoids materializing a separate updated weight matrix for every
token. Matrix products use MLX's Metal backend; `mx.compile` fuses eligible
elementwise operations in the pure inner functions. This is an MLX implementation,
not a port of CUDA/Triton kernels. There is no hand-written Metal kernel here.

With batch `B`, heads `H`, head dimension `D`, and FP32 state, the persistent
fast-weight and gradient storage per layer is:

| Learner | Bytes, excluding convolution history |
| --- | ---: |
| Linear | `8 * B * H * (D² + D)` |
| MLP | `8 * B * H * (8D² + 5D)` |

This does not grow with context length. Temporary activations, outputs, and
unevaluated graphs do consume additional memory. `prefill` and MLX-LM generation
provide evaluation boundaries; see MLX's
[lazy-evaluation documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

Convolution history is gathered into compact buffers so evaluated caches and
snapshots retain only the last `conv_kernel - 1` inputs per convolution. Training
through an unevaluated graph still retains the dependencies needed for gradients.

## Benchmark

```bash
python -m ttt_mlx.benchmark --kind linear
python -m ttt_mlx.benchmark --kind mlp
python -m ttt_mlx.benchmark --kind mlp --eager
```

Reports synchronized prefill/decode throughput, cache bytes, and peak MLX
allocated memory. One warmup is excluded. `--eager` disables the inner compile
wrapper for comparison. Decode uses fixed random token IDs to measure execution.
See [validation.md](docs/validation.md) for measured results and reference revisions.

## Current boundaries

Dynamic batching of different sequence lengths, padding masks, cache rewind /
speculative decoding, distributed training, and custom Metal kernels are not
implemented. Use a fresh cache for each independent request. Speculative decoding
is rejected by MLX-LM because this recurrent cache is not trimmable.

The verification uses random small models and synthetic tokens, including the
reference head dimension 64 and mini-batch size 16. No pretrained TTT checkpoint
was supplied, so perplexity, downstream quality, and billion-parameter performance
are unmeasured. This implementation does not convert an attention-trained LLM
into a trained TTT model.

See [design.md](docs/design.md) for state semantics and the dual-form equations.
The reference project's MIT license is preserved in [LICENSE.reference](LICENSE.reference).
