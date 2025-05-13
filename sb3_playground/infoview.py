from jax import tree_util
import jax.numpy as jnp


class InfoElementView:
    """A dict-like view into a single element of a PyTree info structure."""

    def __init__(self, info, index, scalar_unwrap=False):
        self._info = info
        self._index = index
        self._scalar_unwrap = scalar_unwrap

    def _maybe_unwrap_scalar(self, x):
        if self._scalar_unwrap and hasattr(x, "shape") and x.shape == ():
            return x.item()
        return x

    def __getitem__(self, key):
        if key == "TimeLimit.truncated":
            # Truncated is true when the episode ended due to step limit
            term = self._info.get("termination", None)
            trunc = self._info.get("truncation", None)
            if trunc is not None and term is not None:
                return bool(trunc[self._index] and not term[self._index])
            return False
        elif key == "terminal_observation":
            last_obs = self._info.get("last_obs", None)
            if last_obs is not None:
                return tree_util.tree_map(lambda x: x[self._index], last_obs)
            raise KeyError("terminal_observation not available")
        else:
            value = self._info[key]
            return tree_util.tree_map(lambda x: self._maybe_unwrap_scalar(x[self._index]), value)

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

    def __init__(self, info, scalar_unwrap=True):
        self._info = info
        self._scalar_unwrap = scalar_unwrap

    def __len__(self):
        # Use length of any first-leaf array (assumes uniform length)
        return len(self._info)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds")
        return InfoElementView(self._info, index, scalar_unwrap=self._scalar_unwrap)

    def to_list(self):
        """Convert all elements to a list of plain Python dicts."""
        return [self[i].to_dict() for i in range(len(self))]

    def __repr__(self):
        return f"InfoWrapper(len={len(self)}, scalar_unwrap={self._scalar_unwrap})"
