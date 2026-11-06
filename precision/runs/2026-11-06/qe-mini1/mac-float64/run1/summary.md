## Matrix products, float64

| n | NumPy (Accelerate) | JAX, YNNPACK | JAX, XLA's own dot | Largest relative difference from NumPy |
|---:|---:|---:|---:|---|
| 8 | 3.5 GFLOPS | 0.3 GFLOPS | 0.3 GFLOPS | jax-ynn 1.7e-16, jax-xla-dot 1.7e-16 |
| 64 | 215.1 GFLOPS | 28.1 GFLOPS | 23.9 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 0.0e+00 |
| 512 | 498.2 GFLOPS | 404.5 GFLOPS | 350.8 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 1.4e-15 |
| 1,024 | 558.3 GFLOPS | 426.4 GFLOPS | 428.5 GFLOPS | jax-ynn 2.5e-15, jax-xla-dot 2.7e-15 |
| 4,096 | 542.3 GFLOPS | 383.9 GFLOPS | 483.1 GFLOPS | jax-ynn 6.3e-15, jax-xla-dot 6.1e-15 |

Small products in a recurrence, as QuantEcon/project-benchmarks-jax#4 timed them, µs per product (NumPy's includes Python's dispatch; its bare `np.dot` call in brackets):

| n | NumPy | JAX, YNNPACK | JAX, XLA's own dot |
|---:|---:|---:|---:|
| 8 | 0.657 (0.195) | 0.151 | 0.143 |
| 64 | 4.211 (2.444) | 8.133 | 9.924 |

## The cliff (QuantEcon/project-benchmarks-jax#7), float64, GB/s of the m x m array (warm median)

| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 1.91 | 128.2 (ynn) | 45.8 (xla) | 60.1 (xla) | 162.0 (ynn) |
| 18,000 | 2.41 | 35.0 (ynn) | 45.9 (xla) | 59.8 (xla) | 80.9 (ynn) |

## One value-function kernel, MLX on its CPU device against JAX, float64 (warm median)

| Kernel | n | JAX | MLX | MLX's difference from NumPy |
|---|---:|---:|---:|---|
| stored Bellman step | 2,000 | 0.20 ms | 2.21 ms | 0.0e+00 |
| stored Bellman step | 8,000 | 3.55 ms | 35.83 ms | 0.0e+00 |
| stored Bellman step | 14,000 | 11.98 ms | 109.40 ms | 0.0e+00 |
| P @ V | 1,000 | 4.79 ms | 3.47 ms | 0.0e+00 |
| P @ V | 4,000 | 340.50 ms | 233.57 ms | 0.0e+00 |
