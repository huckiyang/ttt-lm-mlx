# Validation record

Validated locally on 2026-09-09 on an **Apple M4 Max**, macOS 26.0.1,
Python 3.12, MLX 0.31.2, the MLX-LM 0.32.0 checkout, PyTorch 2.11.0,
and Transformers 5.6.2.
MLX tests ran on Metal; PyTorch served as the CPU numerical oracle.

## Results

```
python -m pytest tests -q
75 passed, 2 warnings in 5.96s

ruff check ttt_mlx tests examples
All checks passed!

git diff --check
# clean

python -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir /tmp/ttt-mlx-wheels .
# Successfully built ttt-lm-mlx
```

The two pytest warnings are SWIG dependency deprecations. No parity tests were
skipped. The test fixture adapts Transformers 5's tied-weight metadata; reference
math, cache updates, forward functions, and gradient code remain unchanged.

The synthetic outer-training example also completed on Metal:

```
python -m examples.train
step=0 loss=4.867784
step=1 loss=4.741542
step=2 loss=4.690279
step=3 loss=4.644018
step=4 loss=4.598056
```

| Evidence | Coverage |
| --- | --- |
| `test_model_prefill_decode`, 16 cases | Both learners × Q/K sharing × gate × pre-convolution; full logits, partial prefill, single-token decode, fast weights and accumulators |
| `test_outer_gradients`, 4 cases | Both learners, compiled/uncompiled; every trainable layer parameter and input gradient |
| `test_model_training_gradients`, 2 cases | Both learners; cross-entropy and every full-model parameter gradient with untied head and pre-convolution |
| `test_reference_head_dim_64`, 2 cases | Both learners against PyTorch at head dimension 64, mini-batch size 16, 67-token sequence |
| `test_reference_checkpoint_conversion`, 4 cases | Both learners; sharded safetensors and PyTorch `.bin`, convolution layout, tokenizer, FP32 and FP16 outputs |
| Streaming suite, 24 cases | Irregular chunk splits, causality, kernel sizes 1/4, state isolation and persistence, constant cache bytes, generation, mixed precision, optimizer steps, streaming gradients, bounded prefill, input/config checks |
| Review regressions, 23 cases | Physical convolution history allocation and snapshots (batch sizes 1/2); repeated EOS/config saves with/without tokenizer; runtime EOS updates; full/partial mixed 4/8-bit quantization for both learners and tied/untied heads; dequantization; rejection before writing unsupported activation-quantized models; clipped-coefficient output/input/parameter gradient parity with/without compilation |

The tests use the user's MLX-LM checkout through the public class-loader hook
and `generate_step`. Checkpoint tests use local synthetic tokenizers and do not
download data. A wheel builds successfully without dependency resolution.

## Review fixes

The three review findings have regression coverage in `tests/test_regressions.py`.
For the original 8,192-token, 64-channel convolution probe, retaining an evaluated
history snapshot now holds **768 bytes**, down from **2,113,536 bytes**. The probe
deletes the input and output arrays and measures active MLX allocation; it does
not rely only on the cache's logical `nbytes`.

Loaded EOS IDs `{2, 3}` survive repeated saves with or without a tokenizer.
Other checkpoint metadata and an existing `generation_config.json` also survive.
Quantized weight arrays are checked for exact equality after reload, along with
logit parity, including per-module 4/8-bit settings and unquantized projections.
Unsupported `QQLinear` saves raise before creating the destination.

At exactly zero effective token coefficients, both learners now match the
reference gradient, including the case where the zero coefficient commits a
mini-batch. Tests compare every layer parameter gradient and the input gradient,
in addition to forward outputs and loss.

## Baseline small-model timing

These timing and training-example measurements predate the review fixes and used
the installed MLX-LM 0.31.3 package. The 75-test result above covers the revised code.

Default benchmark settings: random weights, vocabulary 256, width 256, four
heads of dimension 64, two layers, batch one, 256 prompt tokens, mini-batches
of 16, prefill chunks of 128, 32 decode steps. FP32 throughout. Median of
three runs after one warmup; times include graph construction and explicit
evaluation of logits and cache state. Runs were executed sequentially.

| Learner | Inner compile | Prefill tokens/s | Decode tokens/s | Cache bytes | Peak allocated bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear | Yes | 48,897 | 1,316 | 266,240 | 12,770,732 |
| MLP | Yes | 26,238 | 1,302 | 2,117,632 | 34,611,064 |
| MLP | No | 19,031 | 1,001 | 2,117,632 | 27,078,768 |

Commands:

```bash
python -m ttt_mlx.benchmark --kind linear
python -m ttt_mlx.benchmark --kind mlp
python -m ttt_mlx.benchmark --kind mlp --eager
```

The compiled MLP was faster in this run, with higher peak allocation. These
are measurements of a small synthetic workload, not evidence of speedups over
PyTorch/CUDA, trained-model quality, or production serving performance. System
load and power settings can change timings. Peak allocation is MLX's allocator
metric, not total process RSS. Cache byte counts match the formulas in the README.

## Reference revisions

| Checkout | Commit |
| --- | --- |
| `ttt-lm-pytorch` | `cd831db10c8c9a0f6340f02da5613316a8a92b67` |
| `ttt-lm-jax` | `6f529b124c7fb5879b33c06926408b15add1d82f` |
| `ttt-lm-kernels` | `99851e6dcc44060952a5618f0131a5ca6d7f6519` |
| `mlx-lm` | `170a11c58ce7d22a720901a2535adb0c218b871b` |

JAX and CUDA/Triton implementations were inspected as design references, not
executed. Numerical parity is established against the supplied PyTorch code.
No trained checkpoint, corpus-scale pretraining, perplexity evaluation, dynamic
batch scheduler, custom Metal kernel, or billion-parameter benchmark was part
of this validation.
