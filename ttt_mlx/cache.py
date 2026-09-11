"""Per-sequence fast weights, partial-mini-batch gradients, and convolution history."""

import json
from copy import copy

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.models.cache import _BaseCache


class TTTCache(_BaseCache):
    def __init__(self):
        self._arrays = {}
        self.offset = 0

    @property
    def state(self):
        return self._arrays

    @state.setter
    def state(self, value):
        self._arrays = value

    @property
    def meta_state(self):
        return str(self.offset)

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value)

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self._arrays.values())

    def reset(self):
        self._arrays = {}
        self.offset = 0

    def clone(self):
        other = copy(self)
        # Updates replace arrays, so a snapshot shares storage until changed.
        other._arrays = self._arrays.copy()
        return other

    def detach(self):
        self._arrays = tree_map(mx.stop_gradient, self._arrays)


def save_cache(path, caches):
    """Save a nonempty prompt cache, including unfinished mini-batches."""
    if not caches or any(c.empty() for c in caches):
        raise ValueError("Only initialized caches can be saved")
    arrays = dict(tree_flatten([c.state for c in caches]))
    mx.save_safetensors(
        str(path), arrays, {"offsets": json.dumps([c.offset for c in caches])}
    )


def load_cache(path):
    arrays, metadata = mx.load(str(path), return_metadata=True)
    states = tree_unflatten(list(arrays.items()))
    offsets = json.loads(metadata["offsets"])
    if len(states) != len(offsets):
        raise ValueError("Invalid TTT cache file")
    return [
        TTTCache.from_state(state, str(offset))
        for state, offset in zip(states, offsets)
    ]
