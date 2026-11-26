## Matrix products, float64

| n | NumPy (Accelerate) | JAX, YNNPACK | JAX, XLA's own dot | Largest relative difference from NumPy |
|---:|---:|---:|---:|---|
| 8 | 3.5 GFLOPS | 0.3 GFLOPS | 0.3 GFLOPS | jax-ynn 1.7e-16, jax-xla-dot 1.7e-16 |
| 64 | 213.3 GFLOPS | 21.8 GFLOPS | 19.7 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 0.0e+00 |
| 512 | 451.6 GFLOPS | 626.5 GFLOPS | 517.0 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 1.4e-15 |
| 1,024 | 480.2 GFLOPS | 813.2 GFLOPS | 668.0 GFLOPS | jax-ynn 2.5e-15, jax-xla-dot 2.7e-15 |
| 4,096 | 421.2 GFLOPS | 655.2 GFLOPS | 765.7 GFLOPS | jax-ynn 6.3e-15, jax-xla-dot 6.1e-15 |

Small products in a recurrence, as QuantEcon/project-benchmarks-jax#4 timed them, µs per product (NumPy's includes Python's dispatch; its bare `np.dot` call in brackets):

| n | NumPy | JAX, YNNPACK | JAX, XLA's own dot |
|---:|---:|---:|---:|
| 8 | 0.757 (0.218) | 0.175 | 0.175 |
| 64 | 4.607 (2.279) | 8.912 | 11.386 |

## The cliff (QuantEcon/project-benchmarks-jax#7), float64, GB/s of the m x m array (warm median)

| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 1.91 | 208.1 (ynn) | 79.8 (xla) | 125.7 (xla) | 242.4 (ynn) |
| 18,000 | 2.41 | 38.7 (ynn) | 79.6 (xla) | 126.7 (xla) | 78.1 (ynn) |

## One value-function kernel, MLX on its CPU device against JAX, float64 (warm median)

| Kernel | n | JAX | MLX | MLX's difference from NumPy |
|---|---:|---:|---:|---|
| stored Bellman step | 2,000 | 0.17 ms | 2.22 ms | 0.0e+00 |
| stored Bellman step | 8,000 | 2.27 ms | 31.94 ms | 0.0e+00 |
| stored Bellman step | 14,000 | 7.64 ms | 105.77 ms | 0.0e+00 |
| P @ V | 1,000 | 2.55 ms | 3.96 ms | 0.0e+00 |
| P @ V | 4,000 | 203.80 ms | 301.62 ms | 0.0e+00 |
