import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util


@functools.partial(jax.jit, static_argnums=1)
def why(arg, name):
    print(name)


class EpisodeInfoView:
    def __init__(self, info, index, scalar_unwrap=True, to_numpy=True):
        self._info = info
        self._index = index
        self._scalar_unwrap = scalar_unwrap
        self._to_numpy = to_numpy

    def __getitem__(self, key):
        if key == "r":
            key = "episode_return"
        elif key == "l":
            key = "episode_length"
        elif key == "t":
            key = "episode_duration"
        value = tree_util.tree_map(lambda x: x[self._index], self._info[key])
        if self._to_numpy and hasattr(value, "__array__"):
            return np.asarray(value)
        if hasattr(value, "shape") and value.shape == ():
            value = value.item()
        return value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return {"r", "t", "l"}

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return ((k, self[k]) for k in self)

    def to_dict(self):
        """Convert this view to a plain Python dict."""
        return {k: self[k] for k in self}

    def copy(self):
        return self.to_dict()

    # def __repr__(self):
    #     return f"InfoElementView({{ {', '.join(f'{k}: {self[k]!r}' for k in self) } }})"


class InfoElementView:
    """A dict-like view into a single element of a PyTree info structure."""

    def __init__(self, info, index, scalar_unwrap=True, to_numpy=True):
        self._info = info
        self._index = index
        self._scalar_unwrap = scalar_unwrap
        self._to_numpy = to_numpy
        self._get_ep_info = jax.jit(self.get_ep_info)

    @staticmethod
    def get_ep_info(info, index):
        r = tree_util.tree_map(lambda x: x[index], info["episode_return"])  # noqa: E741
        t = tree_util.tree_map(lambda x: x[index], info["episode_duration"])  # noqa: E741
        l = tree_util.tree_map(lambda x: x[index], info["episode_length"])  # noqa: E741
        return r, l, t

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
        elif key == "episode":
            r, l, t = self._get_ep_info(self._info, jnp.int32(self._index))  # noqa: E741
            return dict(r=np.asarray(r).item(), l=np.asarray(l).item(), t=np.asarray(t).item())

            # return EpisodeInfoView(self._info, self._index, self._scalar_unwrap, self._to_numpy)
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
        keys.update({"episode"})
        return keys

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return ((k, self[k]) for k in self)

    def to_dict(self):
        """Convert this view to a plain Python dict."""
        return {k: self[k] for k in self}

    def copy(self):
        return self.to_dict()

    def __repr__(self):
        return f"InfoElementView({{ {', '.join(f'{k}: {self[k]!r}' for k in self) } }})"


class InfoWrapper:
    """A list-like wrapper for a PyTree info dict-of-arrays."""

    def __init__(self, info, scalar_unwrap=True, to_numpy=True, num_envs=None):
        self._info = info
        self._scalar_unwrap = scalar_unwrap
        self._to_numpy = to_numpy
        self._num_envs = num_envs

    def __len__(self):
        # Use length of any first-leaf array (assumes uniform length)
        if self._num_envs:
            return self._num_envs
        return tree_util.tree_leaves(tree_util.tree_map(lambda x: len(x), self._info))[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.to_list(index)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds")
        return InfoElementView(self._info, index, scalar_unwrap=self._scalar_unwrap, to_numpy=self._to_numpy)

    def to_list(self, slicer=slice(None)):
        """Convert all elements to a list of plain Python dicts."""
        return [self[i].to_dict() for i in range(len(self))[slicer]]

    # def to_list(self):
    #     """Convert all elements to a list of plain Python dicts."""
    #     return [self[i].to_dict() for i in range(len(self))]

    def __repr__(self):
        return f"InfoWrapper(len={len(self)}, scalar_unwrap={self._scalar_unwrap}, to_numpy={self._to_numpy})"
