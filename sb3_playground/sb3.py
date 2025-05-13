from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .infoview import InfoWrapper
from .utils import split_rng_key


class Mjx2SB3VecEnv(VecEnv):
    def __init__(self, env, num_envs, rng):
        self.env = env
        self._num_envs = num_envs
        self.rng = rng

        obs_shape = (env.observation_size,)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        action_limits = env.mj_model.actuator_ctrlrange
        self._action_low = np.asarray(action_limits[:, 0])
        self._action_high = np.asarray(action_limits[:, 1])
        self._action_space = gym.spaces.Box(low=self._action_low, high=self._action_high, dtype=np.float32)

        self._reset_fn = jax.jit(jax.vmap(env.reset))
        self._step_fn = jax.jit(jax.vmap(env.step))
        self._replenish_fn = jax.jit(jax.vmap(env.replenish_reset))
        self._render_fn = jax.jit(jax.vmap(env.render))

        self._state = self._reset_fn(self._next_keys())
        self.reset_infos = InfoWrapper(self._state.info)
        self._next_state = None

    def _next_keys(self):
        self.rng, keys = split_rng_key(self.rng, (self._num_envs,))
        return keys

    def reset(self):
        self._state = self._reset_fn(self._next_keys())
        self.reset_infos = InfoWrapper(self._state.info)
        return np.asarray(self._state.obs)

    def step_async(self, actions):
        clipped = np.clip(actions, self._action_low, self._action_high)
        actions = jnp.asarray(clipped)
        self._next_state = self._step_fn(self._state, actions)

    def step_wait(self):
        self._state = self._next_state
        obs = np.asarray(self._state.obs)
        rewards = np.asarray(self._state.reward)
        dones = np.asarray(self._state.done).astype(bool)

        if np.any(dones):
            self._state = self._replenish_fn(self._next_keys(), self._state)

        infos = InfoWrapper(self._state.info)
        return obs, rewards, dones, infos

    def render(self, mode="rgb_array"):
        return np.asarray(self._render_fn(self._state))

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    # -- the following abstract methods need to be there, but I'm not sure yet how to implement them for MjxEnv.
    def get_attr(self, attr_name: str, indices: Optional[Union[int, Iterable[int]]] = None) -> Any:
        """
        Return attribute from the underlying environments.

        Args:
            attr_name (str): The name of the attribute to retrieve.
            indices (Optional[Union[int, Iterable[int]]]): Indices of environments to query.
                If None, applies to all environments.

        Returns:
            Any: The attribute(s) value(s).
        """
        # Example implementation:
        # indices = self._get_indices(indices)
        # return [getattr(self.envs[i], attr_name) for i in indices]
        raise NotImplementedError("get_attr is not implemented in this wrapper.")

    def set_attr(self, attr_name: str, value: Any, indices: Optional[Union[int, Iterable[int]]] = None) -> None:
        """
        Set attribute in the underlying environments.

        Args:
            attr_name (str): The name of the attribute to set.
            value (Any): The value to assign.
            indices (Optional[Union[int, Iterable[int]]]): Indices of environments to modify.
                If None, applies to all environments.
        """
        # Example implementation:
        # indices = self._get_indices(indices)
        # for i in indices:
        #     setattr(self.envs[i], attr_name, value)
        raise NotImplementedError("set_attr is not implemented in this wrapper.")

    def env_method(
        self, method_name: str, *method_args, indices: Optional[Union[int, Iterable[int]]] = None, **method_kwargs
    ) -> Any:
        """
        Call a method on the underlying environments.

        Args:
            method_name (str): The name of the method to call.
            *method_args: Positional arguments to pass.
            indices (Optional[Union[int, Iterable[int]]]): Indices of environments to apply the method to.
                If None, applies to all environments.
            **method_kwargs: Keyword arguments to pass.

        Returns:
            Any: The return value(s) of the method call(s).
        """
        # Example implementation:
        # indices = self._get_indices(indices)
        # return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs) for i in indices]
        raise NotImplementedError("env_method is not implemented in this wrapper.")

    def env_is_wrapped(
        self, wrapper_class: type, indices: Optional[Union[int, Iterable[int]]] = None
    ) -> Union[bool, List[bool]]:
        """
        Check if the environments are wrapped with a given wrapper.

        :param wrapper_class: The class of the wrapper to check.
        :param indices: Indices of the environments to check. If None, check all.
        :return: A boolean or list of booleans indicating if the env(s) are wrapped.
        """
        # Example implementation (commented out):
        # if indices is None:
        #     indices = range(self.num_envs)
        # elif isinstance(indices, int):
        #     indices = [indices]
        # return [isinstance(self.envs[i], wrapper_class) for i in indices]

        raise NotImplementedError("env_is_wrapped is not implemented.")
