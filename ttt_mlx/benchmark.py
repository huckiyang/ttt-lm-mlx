"""Synthetic inference timing; weights and tokens are random, not quality results."""

import argparse
import json
import platform
import statistics
import time

import mlx.core as mx

from .config import ModelArgs
from .model import Model
from .streaming import prefill


def benchmark(args):
    mx.random.seed(0)
    config = ModelArgs(
        hidden_size=args.width,
        intermediate_size=args.width * 3,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        vocab_size=256,
        mini_batch_size=args.mini_batch_size,
        ttt_layer_type=args.kind,
        compile_inner=not args.eager,
    )
    model = Model(config)
    model.eval()
    if args.dtype != "float32":
        model.set_dtype(getattr(mx, args.dtype))
    tokens = mx.random.randint(0, config.vocab_size, (args.batch, args.length))
    next_token = tokens[:, :1]
    mx.eval(model.parameters(), tokens)
    times, decodes = [], []
    for trial in range(args.repeats + 1):
        cache = model.make_cache()
        mx.reset_peak_memory()
        start = time.perf_counter()
        _, cache = prefill(model, tokens, cache, args.chunk_size)
        prefill_time = time.perf_counter() - start
        decode_start = time.perf_counter()
        for _ in range(args.decode):
            logits = model(next_token, cache=cache)
            mx.eval(logits, [c.state for c in cache])
        decode_time = time.perf_counter() - decode_start
        if trial:
            times.append(prefill_time)
            decodes.append(decode_time)
    return {
        "kind": args.kind,
        "compiled": not args.eager,
        "dtype": args.dtype,
        "mlx": mx.__version__,
        "platform": platform.platform(),
        "device": mx.device_info()["device_name"],
        "width": args.width,
        "heads": args.heads,
        "layers": args.layers,
        "batch": args.batch,
        "length": args.length,
        "mini_batch_size": args.mini_batch_size,
        "chunk_size": args.chunk_size,
        "decode_tokens": args.decode,
        "repeats": args.repeats,
        "prefill_tokens_per_second": args.batch
        * args.length
        / statistics.median(times),
        "decode_tokens_per_second": args.batch
        * args.decode
        / statistics.median(decodes),
        "cache_bytes": sum(c.nbytes for c in cache),
        "peak_bytes": mx.get_peak_memory(),
        "note": "Random tiny model; includes graph building and synchronization; excludes warmup.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if min(args.batch, args.length, args.chunk_size, args.decode, args.repeats) < 1:
        parser.error("batch, length, chunk-size, decode, and repeats must be positive")
    print(json.dumps(benchmark(args), indent=2))


if __name__ == "__main__":
    main()
