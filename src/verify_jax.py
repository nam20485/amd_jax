"""Verify that JAX detects and can compute on the AMD ROCm GPU."""

import jax
import jax.numpy as jnp


def main() -> None:
    print("JAX version:      ", jax.__version__)
    print("Default backend:  ", jax.default_backend())
    print("Devices:          ", jax.devices())

    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    y = jnp.arange(1024 * 1024, dtype=jnp.float32).reshape(1024, 1024)

    z = jax.jit(lambda a, b: (a @ b).sum())(x, y)
    z.block_until_ready()

    print("Matmul sum (GPU): ", float(z))
    print("JAX verification OK")


if __name__ == "__main__":
    main()
