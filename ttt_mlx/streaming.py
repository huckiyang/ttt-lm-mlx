import mlx.core as mx


def prefill(model, tokens, cache=None, chunk_size=256):
    """Bound the inference graph and return last-token logits plus the cache.

    Evaluation boundaries keep historical compute graphs out of the resident
    recurrent state. Use model(...) directly when differentiating a sequence.
    """
    if tokens.ndim != 2 or tokens.shape[1] == 0 or chunk_size <= 0:
        raise ValueError(
            "Expected nonempty [batch, sequence] tokens and positive chunk_size"
        )
    cache = model.make_cache() if cache is None else cache
    for start in range(0, tokens.shape[1], chunk_size):
        logits = model(tokens[:, start : start + chunk_size], cache=cache)[:, -1:]
        mx.eval(logits, [c.state for c in cache])
    return logits, cache
