from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import mujoco
import mujoco_playground
import numpy as np

# import mujoco_playground.locomotion
import tqdm
from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper



ENV_NAME = "CartpoleBalance"
# ENV_NAME = "SwimmerSwimmer6"
NUM_ENVS = 128
rng = jax.random.PRNGKey(42)




def tree_slice(pytreenode, the_slice):
    def _slice(x):
        return x[the_slice]

    return jax.tree.map(_slice, pytreenode)


def split_rng_key(rng_key, shape):
    """
    Splits an RNG key into multiple keys based on the given shape,
    plus one additional key.

    Parameters:
    - rng_key: A JAX random key.
    - shape: A tuple indicating the number of keys to generate.

    Returns:
    - new_rng_key: A new JAX random key.
    - split_keys: A JAX numpy array of keys shaped according to `shape`.
    """
    total_keys = jnp.prod(jnp.array(shape)) + 1
    keys = jax.random.split(rng_key, total_keys)
    new_rng_key, split_keys = keys[0], keys[1:]
    split_keys = split_keys.reshape(shape + (2,))
    return new_rng_key, split_keys


class ReplenishableAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done using a cached initial reset obs and data
    stored inside `state.info`. External code can call `replenish_reset(rng, state)` to
    update the resettable data in the info dictionary.
    """

    def __init__(self, env):
        super().__init__(env)

    def replenish_reset(self, rng: jax.Array, state: mjx_env.State) -> mjx_env.State:
        """Returns a new state with updated reset obs/data stored in `info`."""
        reset_state = self.env.reset(rng)
        state.info["reset_obs"] = reset_state.obs
        state.info["reset_data"] = reset_state.data
        return state

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Initial reset also sets up initial reset_obs/reset_data in info."""
        state = self.env.reset(rng)
        state.info["reset_obs"] = state.obs
        state.info["reset_data"] = state.data
        state.info["last_obs"] = state.obs
        state.info["last_data"] = state.data
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Steps the environment and resets done environments using obs/data in info."""
        done_type = state.done.dtype
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        if "termination" in state.info:
            termination = state.info["termination"]
            termination = jnp.where(state.done, jnp.zeros_like(termination), termination)
            state.info.update(termination=termination)
        if "truncation" in state.info:
            truncation = state.info["truncation"]
            truncation = jnp.where(state.done, jnp.zeros_like(truncation), truncation)
            state.info.update(truncation=truncation)

        state = state.replace(done=jnp.zeros_like(state.done, dtype=done_type))
        state = self.env.step(state, action)

        done = state.done

        def where_done(x, y):
            if done.shape:
                reshape_done = jnp.reshape(done, [x.shape[0]] + [1] * (x.ndim - 1))
            else:
                reshape_done = done
            return jnp.where(reshape_done, x, y)

        state.info["last_obs"] = state.obs
        state.info["last_data"] = state.data
        reset_obs = state.info["reset_obs"]
        reset_data = state.info["reset_data"]
        new_obs = jax.tree.map(where_done, reset_obs, state.obs)
        new_data = jax.tree.map(where_done, reset_data, state.data)
        return state.replace(obs=new_obs, data=new_data)


class MjxEpisodeWrapper(mujoco_playground.wrapper.Wrapper):
    """Maintains episode step count and sets both termination and truncation at episode end for mjx_env."""

    def __init__(self, env: mjx_env.MjxEnv, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        batch_shape = rng.shape[:-1]

        done_type = state.done.dtype
        state.info["steps"] = jnp.zeros(batch_shape, dtype=jnp.int32)
        state.info["termination"] = jnp.zeros(batch_shape, dtype=done_type)
        state.info["truncation"] = jnp.zeros(batch_shape, dtype=done_type)

        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        done_type = state.done.dtype

        def f(s, _):
            ns = self.env.step(s, action)
            return ns, ns.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))

        steps = state.info["steps"] + self.action_repeat
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)

        # Determine whether episode is done due to step limit
        done_due_to_steps = steps >= episode_length

        # Update termination and truncation flags in-place
        state.info["termination"] = jnp.where(state.done, 1, 0).astype(done_type)
        state.info["truncation"] = jnp.where(done_due_to_steps, 1 - state.done, 0).astype(done_type)
        state.info["steps"] = steps

        done = jnp.logical_or(state.done, done_due_to_steps).astype(done_type)
        return state.replace(done=done)


class MjxRenderWrapper(mujoco_playground.wrapper.Wrapper):
    """Wrapper that replaces only the render method of an MJX environment, using a jaxwith a custom JAX-compatible version."""

    def render(
        self,
        trajectory: Sequence[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
    ) -> Sequence[np.ndarray]:
        result_shape = jax.ShapeDtypeStruct((height, width, 3), jnp.uint8)
        # Use JAX's pure_callback to perform the rendering computation.
        return jax.pure_callback(
            self.env.render,
            result_shape,
            trajectory,
            height,
            width,
            camera,
            scene_option=scene_option,
            modify_scene_fns=modify_scene_fns,
            vmap_method="sequential",
        )


wenv = MjxRenderWrapper(_env)
jax.vmap(wenv.render)(ustate)


_key, rng = jax.random.split(rng, 2)
keys = jnp.asarray(jax.random.split(_key, NUM_ENVS))
env_cfg = mujoco_playground.registry.get_default_config(ENV_NAME)
env_cfg.episode_length = 3
_env = registry.load(ENV_NAME, config=env_cfg)
tlenv = MjxEpisodeWrapper(_env, episode_length=env_cfg.episode_length, action_repeat=env_cfg.action_repeat)
env = ReplenishableAutoResetWrapper(tlenv)

# env = mujoco_playground.wrapper.BraxAutoResetWrapper(_env)
jitted_reset = jax.jit(jax.vmap(env.reset))
jitted_step = jax.jit(jax.vmap(env.step))
jitted_replenish = jax.jit(jax.vmap(env.replenish_reset))

print("setup done")
action_limits = env.mj_model.actuator_ctrlrange
state0 = jitted_reset(keys)
print("jitted reset done")
action = jnp.array(np.random.uniform(size=(NUM_ENVS, env.action_size)))
state = jitted_step(state0, action)
print("jitted step done")
# env.observation_size

# jitted_step = jax.jit(jax.vmap(env.step))
for i in tqdm.trange(1000):
    state = jitted_step(state, action)
    if any(state.done):
        print("any state was done!")
        rng, keys = split_rng_key(rng, (NUM_ENVS,))
        state = jitted_replenish(keys, state)
