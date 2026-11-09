## Matrix products, float64

| n | NumPy (Accelerate) | JAX, YNNPACK | JAX, XLA's own dot | Largest relative difference from NumPy |
|---:|---:|---:|---:|---|
| 8 | 3.5 GFLOPS | 0.3 GFLOPS | 0.3 GFLOPS | jax-ynn 1.7e-16, jax-xla-dot 1.7e-16 |
| 64 | 220.8 GFLOPS | 25.1 GFLOPS | 23.6 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 0.0e+00 |
| 512 | 501.0 GFLOPS | 398.2 GFLOPS | 344.6 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 1.4e-15 |
| 1,024 | 550.1 GFLOPS | 384.9 GFLOPS | 395.4 GFLOPS | jax-ynn 2.5e-15, jax-xla-dot 2.7e-15 |
| 4,096 | 537.5 GFLOPS | 341.0 GFLOPS | 440.2 GFLOPS | jax-ynn 6.3e-15, jax-xla-dot 6.1e-15 |

Small products in a recurrence, as QuantEcon/project-benchmarks-jax#4 timed them, µs per product (NumPy's includes Python's dispatch; its bare `np.dot` call in brackets):

| n | NumPy | JAX, YNNPACK | JAX, XLA's own dot |
|---:|---:|---:|---:|
| 8 | 0.725 (0.212) | 0.160 | 0.165 |
| 64 | 4.109 (2.438) | 8.381 | 10.348 |

## The cliff (QuantEcon/project-benchmarks-jax#7), float64, GB/s of the m x m array (warm median)

| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 1.91 | 121.8 (ynn) | 45.7 (xla) | 60.0 (xla) | 159.9 (ynn) |
| 18,000 | 2.41 | 34.9 (ynn) | 45.5 (xla) | 59.9 (xla) | 79.8 (ynn) |

## One value-function kernel, MLX on its CPU device against JAX, float64 (warm median)

| Kernel | n | JAX | MLX | MLX's difference from NumPy |
|---|---:|---:|---:|---|
| stored Bellman step | 2,000 | 0.21 ms | 2.22 ms | 0.0e+00 |
| stored Bellman step | 8,000 | 3.58 ms | 36.88 ms | 0.0e+00 |
| stored Bellman step | 14,000 | 12.17 ms | 113.21 ms | 0.0e+00 |
| P @ V | 1,000 | 5.21 ms | 3.63 ms | 0.0e+00 |
| P @ V | 4,000 | 366.98 ms | 234.26 ms | 0.0e+00 |
