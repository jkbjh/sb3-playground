import jax
import jax.numpy as jnp


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


# Function to transpose the structure
def transpose_pytree(pytree):
    # Extract the first dimension length
    first_dim_length = next(iter(pytree.values())).shape[0]

    # Create a list of dictionaries
    transposed = []
    for i in range(first_dim_length):
        # Use tree_map to extract the i-th element from each array
        transposed.append(jax.tree_util.tree_map(lambda x: x[i], pytree))
    return transposed


def dict_copy_without(indict: dict, forbiddenkeys: set) -> dict:
    newdict = dict(indict)
    for badkey in forbiddenkeys:
        newdict.pop(badkey, None)
    return newdict
