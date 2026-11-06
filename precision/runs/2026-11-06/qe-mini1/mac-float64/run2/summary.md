## Matrix products, float64

| n | NumPy (Accelerate) | JAX, YNNPACK | JAX, XLA's own dot | Largest relative difference from NumPy |
|---:|---:|---:|---:|---|
| 8 | 4.1 GFLOPS | 0.3 GFLOPS | 0.3 GFLOPS | jax-ynn 1.7e-16, jax-xla-dot 1.7e-16 |
| 64 | 217.0 GFLOPS | 28.1 GFLOPS | 23.5 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 0.0e+00 |
| 512 | 498.0 GFLOPS | 383.7 GFLOPS | 347.8 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 1.4e-15 |
| 1,024 | 557.1 GFLOPS | 434.2 GFLOPS | 429.5 GFLOPS | jax-ynn 2.5e-15, jax-xla-dot 2.7e-15 |
| 4,096 | 544.7 GFLOPS | 378.5 GFLOPS | 468.0 GFLOPS | jax-ynn 6.3e-15, jax-xla-dot 6.1e-15 |

Small products in a recurrence, as QuantEcon/project-benchmarks-jax#4 timed them, µs per product (NumPy's includes Python's dispatch; its bare `np.dot` call in brackets):

| n | NumPy | JAX, YNNPACK | JAX, XLA's own dot |
|---:|---:|---:|---:|
| 8 | 0.674 (0.203) | 0.151 | 0.144 |
| 64 | 4.147 (2.432) | 8.107 | 10.249 |

## The cliff (QuantEcon/project-benchmarks-jax#7), float64, GB/s of the m x m array (warm median)

| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 1.91 | 132.7 (ynn) | 45.9 (xla) | 59.9 (xla) | 162.0 (ynn) |
| 18,000 | 2.41 | 35.2 (ynn) | 46.2 (xla) | 60.4 (xla) | 80.9 (ynn) |

## One value-function kernel, MLX on its CPU device against JAX, float64 (warm median)

| Kernel | n | JAX | MLX | MLX's difference from NumPy |
|---|---:|---:|---:|---|
| stored Bellman step | 2,000 | 0.21 ms | 2.21 ms | 0.0e+00 |
| stored Bellman step | 8,000 | 3.53 ms | 36.09 ms | 0.0e+00 |
| stored Bellman step | 14,000 | 12.03 ms | 110.72 ms | 0.0e+00 |
| P @ V | 1,000 | 4.70 ms | 3.19 ms | 0.0e+00 |
| P @ V | 4,000 | 340.93 ms | 232.59 ms | 0.0e+00 |
