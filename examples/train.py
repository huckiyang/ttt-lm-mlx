"""Small outer-training example. Replace the synthetic token batch with your data."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ttt_mlx import Model, ModelArgs

mx.random.seed(0)
model = Model(
    ModelArgs(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=128,
        ttt_layer_type="mlp",
    )
)
optimizer = optim.Adam(learning_rate=1e-4)
tokens = mx.random.randint(0, 128, (2, 33))


def loss_fn(model, tokens):
    logits = model(tokens[:, :-1])
    return nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="mean")


value_and_grad = nn.value_and_grad(model, loss_fn)
for step in range(5):
    loss, grads = value_and_grad(model, tokens)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
    print(f"step={step} loss={loss.item():.6f}")
