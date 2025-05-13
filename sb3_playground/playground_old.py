# pprint(env_cfg)
# mujoco_playground.registry.get_default_config

# env_cfg
# env_cfg.vision_config
# env_cfg.vision_config.render_batch_size = 3

# env.reset(keys)
# keys.shape
# env.reset(keys.T)
# sys.__file__
# tqdm.__file__
# env
# env.actino_space

# env.reset()


# rng = jax.random.PRNGKey(42)
# rng
# env.reset(rng)
# state = env.reset(rng)
# rng.split()
# rng
# rng = jax.random.PRNGKey(42)
# rng
# type(rng)
# jax.random.split(rng)
# jax.random.split(rng, 2)
# jax.random.split(rng, 3)
# jax.random.split(rng, 10)
# env.reset(jax.random.split(rng, 10))

# keys = jnp.asarray(jax.random.split(rng, 3))
# keys
# env.reset(keys)
# env.reset(keys, 3)
# jax.jit(env.reset)(keys, 3)
# env_cfg = mujoco_playground.registry.get_default_config(ENV_NAME)


# pprint(env_cfg)
# keys = jnp.asarray(jax.random.split(rng, 3))
# env_cfg
# env_cfg.vision_config
# env_cfg.vision_config.render_batch_size = 3
# env_cfg
# env = registry.load("CartpoleBalance", config=env_cfg)
# env.reset(keys)
# keys.shape
# env.reset(keys.T)
# env.reset(keys)
# env_cfg


# class AutoResetWrapper(Wrapper):
#     """Automatically resets Brax envs that are done, for stable-baselines3."""

#     def reset(self, rng: jnp.ndarray) -> mjx_env.State:
#         state = self.env.reset(rng)
#         state.info["first_pipeline_state"] = state.pipeline_state
#         state.info["first_obs"] = state.obs
#         return state

#     def step(self, state: mjx_env.State, action: jnp.ndarray) -> mjx_env.State:
#         if "steps" in state.info:
#             steps = state.info["steps"]
#             steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
#             state.info.update(steps=steps)

#         # Save terminal observation before the step
#         terminal_obs = state.obs

#         state = state.replace(done=jnp.zeros_like(state.done))
#         state = self.env.step(state, action)

#         def where_done(x, y):
#             done = state.done
#             if done.shape:
#                 done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
#             return jnp.where(done, x, y)

#         pipeline_state = jax.tree_map(where_done, state.info["first_pipeline_state"], state.pipeline_state)
#         obs = where_done(state.info["first_obs"], state.obs)

#         # Add terminal observation for bootstrap
#         state.info["terminal_observation"] = terminal_obs

#         # Check time truncation
#         state.info["TimeLimit.truncated"] = jnp.where(
#             steps >= self.env.episode_length, 1 - state.done, jnp.zeros_like(state.done)
#         )

#         return state.replace(pipeline_state=pipeline_state, obs=obs)


# from brax.envs.base import Env, mjx_env.State, Wrapper
# mujoco_playground.wrapper.BraxAutoResetWrapper(env)
# class EpisodeWrapper(Wrapper):
#   """Maintains episode step count and sets done at episode end."""

#   def __init__(self, env: Env, episode_length: int, action_repeat: int):
#     super().__init__(env)
#     self.episode_length = episode_length
#     self.action_repeat = action_repeat

#   def reset(self, rng: jnp.ndarray) -> mjx_env.State:
#     state = self.env.reset(rng)
#     state.info['steps'] = jnp.zeros(rng.shape[:-1])
#     state.info['truncation'] = jnp.zeros(rng.shape[:-1])
#     return state

#   def step(self, state: mjx_env.State, action: jnp.ndarray) -> mjx_env.State:
#     def f(state, _):
#       nstate = self.env.step(state, action)
#       return nstate, nstate.reward

#     state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
#     state = state.replace(reward=jnp.sum(rewards, axis=0))
#     steps = state.info['steps'] + self.action_repeat
#     one = jnp.ones_like(state.done)
#     zero = jnp.zeros_like(state.done)
#     episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
#     done = jnp.where(steps >= episode_length, one, state.done)
#     state.info['truncation'] = jnp.where(
#         steps >= episode_length, 1 - state.done, zero
#     )
#     state.info['steps'] = steps
#     return state.replace(done=done)

# class ReplenishableAutoResetWrapper(Wrapper):
#     """Automatically resets Brax envs that are done using a cached initial reset state.

#     Uses `self.env.reset` for replenishing. External code can call `replenish_reset(rng)`
#     to update the cached reset state.
#     """

#     def __init__(self, env):
#         super().__init__(env)
#         self._initial_state = None  # Will hold the cached reset state

#     def replenish_reset(self, rng: jax.Array):
#         """Updates the cached reset state using `self.env.reset`."""
#         self._initial_state = self.env.reset(rng)

#     def reset(self, rng: jax.Array) -> mjx_env.State:
#         state = self.env.reset(rng)
#         # Set the stored reset state to the first state, if not already set
#         if self._initial_state is None:
#             self._initial_state = state
#         return state

#     def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
#         """Steps the environment and resets done environments using cached initial state."""
#         if "steps" in state.info:
#             steps = state.info["steps"]
#             steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
#             state.info.update(steps=steps)

#         # Step the environment (forcefully set done=False before stepping)
#         state = state.replace(done=jnp.zeros_like(state.done))
#         state = self.env.step(state, action)

#         done = state.done

#         def where_done(x, y):
#             if done.shape:
#                 reshape_done = jnp.reshape(done, [x.shape[0]] + [1] * (x.ndim - 1))
#             else:
#                 reshape_done = done
#             return jnp.where(reshape_done, x, y)

#         new_data = jax.tree.map(where_done, self._initial_state.data, state.data)
#         new_obs = jax.tree.map(where_done, self._initial_state.obs, state.obs)
#         return state.replace(data=new_data, obs=new_obs)


# class ReplenishableAutoResetWrapper(Wrapper):
#     """Automatically resets Brax envs that are done using a cached initial reset observation and data.

#     Uses `self.env.reset` for replenishing. External code can call `replenish_reset(rng)`
#     to update the cached reset state.
#     """

#     def __init__(self, env):
#         super().__init__(env)
#         self._initial_obs = None
#         self._initial_data = None

#     def replenish_reset(self, rng: jax.Array):
#         """Updates the cached reset obs and data using `self.env.reset`."""
#         reset_state = self.env.reset(rng)
#         self._initial_obs = reset_state.obs
#         self._initial_data = reset_state.data

#     def reset(self, rng: jax.Array) -> mjx_env.State:
#         state = self.env.reset(rng)
#         # Set the stored reset obs/data if not already set
#         if self._initial_obs is None or self._initial_data is None:
#             self._initial_obs = state.obs
#             self._initial_data = state.data
#         return state

#     def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
#         """Steps the environment and resets done environments using cached initial obs/data."""
#         if "steps" in state.info:
#             steps = state.info["steps"]
#             steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
#             state.info.update(steps=steps)

#         # Clear 'done' before stepping
#         state = state.replace(done=jnp.zeros_like(state.done))
#         state = self.env.step(state, action)

#         done = state.done

#         def where_done(x, y):
#             if done.shape:
#                 reshape_done = jnp.reshape(done, [x.shape[0]] + [1] * (x.ndim - 1))
#             else:
#                 reshape_done = done
#             return jnp.where(reshape_done, x, y)

#         if self._initial_obs is None or self._initial_data is None:
#             raise ValueError("Initial obs/data not set. Call `reset()` or `replenish_reset()` first.")

#         new_data = jax.tree.map(where_done, self._initial_data, state.data)
#         new_obs = jax.tree.map(where_done, self._initial_obs, state.obs)
#         return state.replace(data=new_data, obs=new_obs)

# class ReplenishableAutoResetWrapper(Wrapper):
#     """Automatically resets Brax envs that are done using a cached initial reset observation and data.

#     Uses `self.env.reset` for replenishing. External code can call `replenish_reset(rng)`
#     to update the cached reset state.
#     """

#     def __init__(self, env):
#         super().__init__(env)
#         self._initial_obs = None
#         self._initial_data = None

#     def replenish_reset(self, rng: jax.Array):
#         """Updates the cached reset obs and data using `self.env.reset`."""
#         reset_state = self.env.reset(rng)
#         self._initial_obs = reset_state.obs
#         self._initial_data = reset_state.data

#     def reset(self, rng: jax.Array) -> mjx_env.State:
#         assert self._initial_obs is not None
#         state = self.env.reset(rng)
#         # Set the stored reset obs/data if not already set
#         if self._initial_obs is None or self._initial_data is None:
#             self._initial_obs = state.obs
#             self._initial_data = state.data
#         return state

#     def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
#         """Steps the environment and resets done environments using cached initial obs/data."""
#         if "steps" in state.info:
#             steps = state.info["steps"]
#             steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
#             state.info.update(steps=steps)

#         # Clear 'done' before stepping
#         state = state.replace(done=jnp.zeros_like(state.done))
#         state = self.env.step(state, action)

#         done = state.done

#         def where_done(x, y):
#             if done.shape:
#                 reshape_done = jnp.reshape(done, [x.shape[0]] + [1] * (x.ndim - 1))
#             else:
#                 reshape_done = done
#             return jnp.where(reshape_done, x, y)

#         if self._initial_obs is None or self._initial_data is None:
#             raise ValueError("Initial obs/data not set. Call `reset()` or `replenish_reset()` first.")

#         new_data = jax.tree.map(where_done, self._initial_data, state.data)
#         new_obs = jax.tree.map(where_done, self._initial_obs, state.obs)
#         return state.replace(data=new_data, obs=new_obs)


    
class MjxEpisodeWrapper(mujoco_playground.wrapper.Wrapper):
    """Maintains episode step count and sets both termination and truncation at episode end for mjx_env."""

    def __init__(self, env: mjx_env.MjxEnv, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        state.info["steps"] = jnp.zeros(rng.shape[:-1])
        state.info["termination"] = jnp.zeros(rng.shape[:-1])
        state.info["truncation"] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))

        steps = state.info["steps"] + self.action_repeat
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)

        # Determine if the current episode has reached its length
        done_due_to_steps = steps >= episode_length

        # Update done, termination, and truncation flags
        state.info["termination"] = jnp.where(state.done, jnp.ones_like(state.done), jnp.zeros_like(state.done))
        state.info["truncation"] = jnp.where(done_due_to_steps, 1 - state.done, jnp.zeros_like(state.done))
        done = jnp.logical_or(state.done, done_due_to_steps)
        state.info["steps"] = steps
        return state.replace(done=done)

