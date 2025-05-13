from stable_baselines3.common.vec_env import VecEnv
import gym
import numpy as np
import jax
import jax.numpy as jnp
from infoview import InfoWrapper, InfoElementView

# action_limits = env.mj_model.actuator_ctrlrange
# env.action_size
# env.observation_size

class Mjx2SB3VecEnv(VecEnv):
    def __init__(self, env, num_envs, action_space, key):
        self.env = env
        self.num_envs = num_envs
        self.obs_shape = (env.observation_size,)
        self._action_space = action_space
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self._keys = jnp.asarray(jax.random.split(key, num_envs))
        self._reset = jax.jit(jax.vmap(self.env.reset))
        self._step = jax.jit(jax.vmap(self.env.step))
        self._replenish = jax.jit(jax.vmap(self.env.replenish_reset))
        self._state = self._reset(self._keys)
        self.reset_infos = [{} for _ in range(num_envs)]

    def reset(self):
        self._keys = jnp.asarray(jax.random.split(jax.random.PRNGKey(np.random.randint(0, 1e6)), self.num_envs))
        self._state = self._reset(self._keys)
        obs = np.asarray(self._state.obs)
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return obs

    def step_async(self, actions):
        self._actions = jnp.array(actions)

    def step_wait(self):
        self._state = self._step(self._state, self._actions)
        obs = np.asarray(self._state.obs)
        rewards = np.asarray(self._state.reward)
        dones = np.asarray(self._state.done).astype(bool)
        infos = []

        for i in range(self.num_envs):
            info = {}
            if dones[i]:
                info["TimeLimit.truncated"] = bool(self._state.info["truncation"][i])
                info["terminal_observation"] = np.asarray(self._state.info["last_obs"][i])
            infos.append(info)

        # Replenish after terminal envs
        if np.any(dones):
            replenish_keys = jnp.asarray(jax.random.split(jax.random.PRNGKey(np.random.randint(0, 1e6)), self.num_envs))
            self._state = self._replenish(replenish_keys, self._state)

        return obs, rewards, dones, infos

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        raise NotImplementedError("Rendering is not implemented.")

    def seed(self, seed=None):
        if seed is not None:
            self._keys = jnp.asarray(jax.random.split(jax.random.PRNGKey(seed), self.num_envs))

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
