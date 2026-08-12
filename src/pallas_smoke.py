"""Pallas smoke test: a tiled vector-add kernel on the ROCm GPU."""

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.triton as pltriton

BLOCK_SIZE = 256


def vector_add_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] + y_ref[...]


def vector_add(x: jax.Array, y: jax.Array) -> jax.Array:
    n = x.shape[0]

    if n % BLOCK_SIZE != 0:
        raise ValueError("n must be divisible by BLOCK_SIZE")

    grid = (n // BLOCK_SIZE,)

    block_spec = pl.BlockSpec(
        block_shape=(BLOCK_SIZE,),
        index_map=lambda i: (i,),
    )

    return pl.pallas_call(
        vector_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=[block_spec, block_spec],
        out_specs=block_spec,
        compiler_params=pltriton.CompilerParams(),
        name="vector_add",
    )(x, y)


def main() -> None:
    print("JAX devices:", jax.devices())

    x = jnp.ones(8192, dtype=jnp.float32)
    y = jnp.arange(8192, dtype=jnp.float32)

    z = jax.jit(vector_add)(x, y)
    z.block_until_ready()

    print(z[:8])
    print("Pallas smoke test completed")


if __name__ == "__main__":
    main()
