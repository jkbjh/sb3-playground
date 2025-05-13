from stable_baselines3.common.vec_env import VecEnv
import gym
import numpy as np
import jax
import jax.numpy as jnp
from infoview import InfoWrapper
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
        self._actions = None

    def _next_keys(self):
        self.rng, keys = split_rng_key(self.rng, (self._num_envs,))
        return keys

    def reset(self):
        self._state = self._reset_fn(self._next_keys())
        self.reset_infos = InfoWrapper(self._state.info)
        return np.asarray(self._state.obs)

    def step_async(self, actions):
        clipped = np.clip(actions, self._action_low, self._action_high)
        self._actions = jnp.asarray(clipped)

    def step_wait(self):
        self._state = self._step_fn(self._state, self._actions)
        obs = np.asarray(self._state.obs)
        rewards = np.asarray(self._state.reward)
        dones = np.asarray(self._state.done).astype(bool)

        if np.any(dones):
            self._state = self._replenish_fn(self._next_keys(), self._state)

        infos = InfoWrapper(self._state.info)
        return obs, rewards, dones, infos

    def render(self, mode="rgb_array"):
        return self._render_fn(self._state)

    def seed(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            self.rng, self._keys = split_rng_key(self.rng, (self._num_envs,))

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self._num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)] * self._num_envs

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
