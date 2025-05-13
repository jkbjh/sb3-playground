import jax.numpy as jnp
import numpy as np
from jax import tree_util


class InfoElementView:
    """A dict-like view into a single element of a PyTree info structure."""

    def __init__(self, info, index, scalar_unwrap=True, to_numpy=True):
        self._info = info
        self._index = index
        self._scalar_unwrap = scalar_unwrap
        self._to_numpy = to_numpy

    def _maybe_convert(self, x):
        if self._scalar_unwrap and hasattr(x, "shape") and x.shape == ():
            x = x.item()
        if self._to_numpy and hasattr(x, "__array__"):
            return np.asarray(x)
        return x

    def __getitem__(self, key):
        dtype = None
        if key == "TimeLimit.truncated":
            key = "truncation"
            dtype = jnp.bool_
        elif key == "terminal_observation":
            key = "last_obs"
        value = tree_util.tree_map(lambda x: x[self._index], self._info[key])
        if dtype:
            value = value.astype(dtype)
        if self._scalar_unwrap and hasattr(value, "shape") and value.shape == ():
            value = value.item()
        if self._to_numpy and hasattr(value, "__array__"):
            return np.asarray(value)
        return value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        keys = set(self._info.keys())
        keys.update({"TimeLimit.truncated", "terminal_observation"})
        return keys

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return ((k, self[k]) for k in self)

    def to_dict(self):
        """Convert this view to a plain Python dict."""
        return {k: self[k] for k in self}

    def __repr__(self):
        return f"InfoElementView({{ {', '.join(f'{k}: {self[k]!r}' for k in self) } }})"


class InfoWrapper:
    """A list-like wrapper for a PyTree info dict-of-arrays."""

    def __init__(self, info, scalar_unwrap=True, to_numpy=True):
        self._info = info
        self._scalar_unwrap = scalar_unwrap
        self._to_numpy = to_numpy

    def __len__(self):
        # Use length of any first-leaf array (assumes uniform length)
        return len(self._info)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds")
        return InfoElementView(self._info, index, scalar_unwrap=self._scalar_unwrap, to_numpy=self._to_numpy)

    def to_list(self):
        """Convert all elements to a list of plain Python dicts."""
        return [self[i].to_dict() for i in range(len(self))]

    def __repr__(self):
        return f"InfoWrapper(len={len(self)}, scalar_unwrap={self._scalar_unwrap}, to_numpy={self._to_numpy})"
