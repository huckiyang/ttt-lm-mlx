# Implementation notes

## Reference mapping

| Reference | MLX implementation |
| --- | --- |
| PyTorch `TTTConfig` | `config.ModelArgs` with MLX-LM `BaseModelArgs` |
| `TTTBase`, `TTTLinear`, `TTTMLP` | `layer.TTTLayer` and `functional` |
| `TTTCache` per-model dictionaries | One `cache.TTTCache` per layer |
| `Block`, `TTTModel`, `TTTForCausalLM` | `model.Block`, `Backbone`, `Model` |
| PyTorch `Conv1d` `[out, in/group, kernel]` | MLX `[out, kernel, in/group]` |
| JAX dual mini-batch update | Batched MLX matmuls and triangular coefficients |
| Kernel prefill/decode separation | Same dual function handles whole or partial mini-batches |

The MLX implementation uses adjacent rotary pairs, which is equivalent to the
PyTorch reference's permute / rotate-half / inverse-permute. Positions restart
at each TTT mini-batch. The outer SwiGLU uses SiLU; inner MLP and output gating
use the tanh GELU approximation.

## Fast-state invariant

For each layer and independent sequence, cache `W1`, `b1` (and `W2`, `b2`) are
the weights **at the beginning of the current mini-batch**. Corresponding
`*_grad` arrays sum the learning-rate-weighted gradients of tokens already seen
in that mini-batch. `offset` counts all consumed tokens, including a partial
mini-batch. All heads and rows of one cache share this phase.

For token `i`, the update coefficient is

```
lr_i = ttt_base_lr / head_dim * sigmoid(x_i @ lr_weight + lr_bias)
token_i = max(1 / (i + 1) + learned_token_i, 0)
W_read_i = W_start - token_i * (G_previous + sum_{j <= i} lr_j * grad_W_j)
```

Here `i` is the position within the current mini-batch, not the global sequence.
Inner gradients are evaluated at `W_start`, including on later decode calls.
Using the most recently read weights as the next gradient's base would change
the learning algorithm. Only the final token of a full mini-batch commits the
new weights and clears accumulators. There is no need to pad or commit a partial
mini-batch prematurely.

At an exactly zero effective token scale, the derivative with respect to the
learned scale is one, matching PyTorch `clamp_min`. The implementation uses an
explicit `where(coefficient >= 0, coefficient, 0)` because MLX `maximum` chooses
a different derivative at equality.

## Partial-chunk dual form

For a linear transform with accumulated gradients `G_W`, `G_b`, key rows `K`,
query rows `Q`, gradient rows `g`, column token scales `t`, and key learning
rates `lr`, define the lower-triangular matrix `E_ij = t_i * lr_j` for `j <= i`.
Within one chunk that stays inside a mini-batch:

```
Z_read = Q W + b - t * (Q G_W + G_b) - (E * (Q Kᵀ)) g - E g
G_W_new = G_W + (K * lr)ᵀ g
G_b_new = G_b + sum(lr * g)
```

The term involving previous accumulators allows arbitrary chunk splits. This
avoids the `[batch, heads, tokens, in_features, out_features]` intermediate of
the reference primal partial-update path. The Python driver splits a call at
every mini-batch boundary, commits completed updates, then continues.

TTT-MLP applies the same formula twice. Its first-layer key gradient is
backpropagated through the **starting** second-layer weight. Its second query
representation is GELU of the **adapted first-layer read**. Both weight updates
use gradients from the same starting inner model, as in the reference.

The output is `Q + LayerNorm(Z_read)`, then the full-width post-norm, optional
GELU gate, and output projection. LayerNorm uses population variance and
epsilon `1e-6`. Inner fast-state math uses FP32 unless explicitly configured
otherwise; changing `state_dtype` can change long-sequence numerical stability.

## Execution and training

The state is request-owned and replaced functionally. There are no in-place
updates of trainable model weights and no stop-gradient in the inner equations.
`mx.compile` receives all arrays as explicit arguments, including norm weights,
so updates to outer parameters are visible and differentiable.

`prefill` bounds inference graphs by evaluating after configurable chunks.
`model(...)` does not evaluate internally, preserving ordinary MLX transforms.
The cache exposes `state`, `meta_state`, `offset`, `nbytes`, `empty`, `size`, and
`is_trimmable` for MLX-LM's single-request generation contract. A snapshot uses
immutable array sharing; reset and continuation replace only its own dictionary.

Convolution history uses a gather to materialize the final `kernel - 1` rows.
A simple slice can share the entire prefill allocation, even after evaluation;
its logical `nbytes` would then undercount retained memory. Regression tests
measure active allocation after deleting the full inputs and retaining a snapshot.

Checkpoint JSON is stored outside the model's array/state tree when loading.
Saving merges that metadata with the current model arguments and derives
quantization settings from the actual modules. This preserves EOS IDs and
per-module quantization while removing stale settings after dequantization.

An adapted state cannot be rewound by dropping its last token. Cache trimming
therefore remains disabled. Variable-phase batch merging also needs an explicit
algorithm and is not exposed by this cache.
