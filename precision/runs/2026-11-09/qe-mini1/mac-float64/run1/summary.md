## Matrix products, float64

| n | NumPy (Accelerate) | JAX, YNNPACK | JAX, XLA's own dot | Largest relative difference from NumPy |
|---:|---:|---:|---:|---|
| 8 | 3.5 GFLOPS | 0.1 GFLOPS | 0.3 GFLOPS | jax-ynn 1.7e-16, jax-xla-dot 1.7e-16 |
| 64 | 233.0 GFLOPS | 26.5 GFLOPS | 23.5 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 0.0e+00 |
| 512 | 501.1 GFLOPS | 398.4 GFLOPS | 355.9 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 1.4e-15 |
| 1,024 | 543.5 GFLOPS | 415.7 GFLOPS | 391.3 GFLOPS | jax-ynn 2.5e-15, jax-xla-dot 2.7e-15 |
| 4,096 | 542.8 GFLOPS | 351.2 GFLOPS | 444.2 GFLOPS | jax-ynn 6.3e-15, jax-xla-dot 6.1e-15 |

Small products in a recurrence, as QuantEcon/project-benchmarks-jax#4 timed them, µs per product (NumPy's includes Python's dispatch; its bare `np.dot` call in brackets):

| n | NumPy | JAX, YNNPACK | JAX, XLA's own dot |
|---:|---:|---:|---:|
| 8 | 0.720 (0.211) | 0.159 | 0.149 |
| 64 | 4.111 (2.439) | 8.432 | 10.346 |

## The cliff (QuantEcon/project-benchmarks-jax#7), float64, GB/s of the m x m array (warm median)

| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 1.91 | 122.8 (ynn) | 45.7 (xla) | 59.2 (xla) | 159.9 (ynn) |
| 18,000 | 2.41 | 35.1 (ynn) | 45.5 (xla) | 24.4 (xla) | 79.2 (ynn) |

## One value-function kernel, MLX on its CPU device against JAX, float64 (warm median)

| Kernel | n | JAX | MLX | MLX's difference from NumPy |
|---|---:|---:|---:|---|
| stored Bellman step | 2,000 | 0.21 ms | 2.21 ms | 0.0e+00 |
| stored Bellman step | 8,000 | 3.70 ms | 36.81 ms | 0.0e+00 |
| stored Bellman step | 14,000 | 12.17 ms | 113.00 ms | 0.0e+00 |
| P @ V | 1,000 | 5.06 ms | 3.34 ms | 0.0e+00 |
| P @ V | 4,000 | 363.07 ms | 232.83 ms | 0.0e+00 |
