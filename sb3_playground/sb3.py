from stable_baselines3.common.vec_env import VecEnv
import gym
import numpy as np
import jax
import jax.numpy as jnp
from infoview import InfoWrapper
from .utils import split_rng_key

# action_limits = env.mj_model.actuator_ctrlrange
# env.action_size
# env.observation_size


class Mjx2SB3VecEnv(VecEnv):
    def __init__(self, env, num_envs, rng):
        self.env = env
        self._num_envs = num_envs
        self.rng, self._keys = split_rng_key(rng, (num_envs,))

        # Observation and action space
        obs_shape = (env.observation_size,)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        action_limits = env.mj_model.actuator_ctrlrange
        self._action_low = np.asarray(action_limits[:, 0])
        self._action_high = np.asarray(action_limits[:, 1])
        self._action_space = gym.spaces.Box(low=self._action_low, high=self._action_high, dtype=np.float32)

        # JIT batched functions
        self._reset_fn = jax.jit(jax.vmap(env.reset))
        self._step_fn = jax.jit(jax.vmap(env.step))
        self._replenish_fn = jax.jit(jax.vmap(env.replenish_reset))
        self._render_fn = jax.jit(jax.vmap(env.render))

        # Initial state
        self._state = self._reset_fn(self._keys)
        self.reset_infos = InfoWrapper(self._state.info)
        self._actions = None

    def reset(self):
        self.rng, self._keys = split_rng_key(self.rng, (self._num_envs,))
        self._state = self._reset_fn(self._keys)
        self.reset_infos = InfoWrapper(self._state.info)
        return np.asarray(self._state.obs)

    def step_async(self, actions):
        actions = np.asarray(actions)
        clipped_actions = np.clip(actions, self._action_low, self._action_high)
        self._actions = jnp.asarray(clipped_actions)
        self._next_state = self._step_fn(self._state, self._actions)

    def step_wait(self):
        self._state = self._next_state
        obs = np.asarray(self._state.obs)
        rewards = np.asarray(self._state.reward)
        dones = np.asarray(self._state.done).astype(bool)

        state_info = self._state.info

        # Add SB3 aliases directly into info
        terminations = np.asarray(state_info["termination"]).astype(bool)
        truncations = np.asarray(state_info["truncation"]).astype(bool)
        time_limit_flags = truncations & ~terminations

        state_info["TimeLimit.truncated"] = time_limit_flags
        state_info["terminal_observation"] = state_info["last_obs"]

        infos = InfoWrapper(state_info)

        # Auto-reset if needed
        if np.any(dones):
            self.rng, replenish_keys = split_rng_key(self.rng, (self._num_envs,))
            self._state = self._replenish_fn(replenish_keys, self._state)

        return obs, rewards, dones, infos

    def render(self, mode="rgb_array"):
        return self._render_fn(self._state)

    def seed(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
            self.rng, self._keys = split_rng_key(self.rng, (self._num_envs,))

    def close(self):
        pass

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
