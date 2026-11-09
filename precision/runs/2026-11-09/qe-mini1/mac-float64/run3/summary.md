## Matrix products, float64

| n | NumPy (Accelerate) | JAX, YNNPACK | JAX, XLA's own dot | Largest relative difference from NumPy |
|---:|---:|---:|---:|---|
| 8 | 3.5 GFLOPS | 0.3 GFLOPS | 0.3 GFLOPS | jax-ynn 1.7e-16, jax-xla-dot 1.7e-16 |
| 64 | 244.4 GFLOPS | 26.4 GFLOPS | 24.1 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 0.0e+00 |
| 512 | 499.0 GFLOPS | 395.6 GFLOPS | 344.0 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 1.4e-15 |
| 1,024 | 543.8 GFLOPS | 335.4 GFLOPS | 378.9 GFLOPS | jax-ynn 2.5e-15, jax-xla-dot 2.7e-15 |
| 4,096 | 536.0 GFLOPS | 340.0 GFLOPS | 437.5 GFLOPS | jax-ynn 6.3e-15, jax-xla-dot 6.1e-15 |

Small products in a recurrence, as QuantEcon/project-benchmarks-jax#4 timed them, µs per product (NumPy's includes Python's dispatch; its bare `np.dot` call in brackets):

| n | NumPy | JAX, YNNPACK | JAX, XLA's own dot |
|---:|---:|---:|---:|
| 8 | 0.742 (0.218) | 0.162 | 0.178 |
| 64 | 4.119 (2.439) | 8.386 | 10.345 |

## The cliff (QuantEcon/project-benchmarks-jax#7), float64, GB/s of the m x m array (warm median)

| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 1.91 | 126.2 (ynn) | 45.7 (xla) | 60.1 (xla) | 160.8 (ynn) |
| 18,000 | 2.41 | 34.6 (ynn) | 45.8 (xla) | 58.9 (xla) | 79.7 (ynn) |

## One value-function kernel, MLX on its CPU device against JAX, float64 (warm median)

| Kernel | n | JAX | MLX | MLX's difference from NumPy |
|---|---:|---:|---:|---|
| stored Bellman step | 2,000 | 0.21 ms | 2.22 ms | 0.0e+00 |
| stored Bellman step | 8,000 | 3.60 ms | 36.92 ms | 0.0e+00 |
| stored Bellman step | 14,000 | 12.25 ms | 113.58 ms | 0.0e+00 |
| P @ V | 1,000 | 5.32 ms | 3.72 ms | 0.0e+00 |
| P @ V | 4,000 | 364.39 ms | 234.77 ms | 0.0e+00 |
