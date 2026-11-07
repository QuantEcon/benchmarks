## Matrix products, float64

| n | NumPy (Accelerate) | JAX, YNNPACK | JAX, XLA's own dot | Largest relative difference from NumPy |
|---:|---:|---:|---:|---|
| 8 | 3.1 GFLOPS | 0.3 GFLOPS | 0.3 GFLOPS | jax-ynn 1.7e-16, jax-xla-dot 1.7e-16 |
| 64 | 206.3 GFLOPS | 21.9 GFLOPS | 19.8 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 0.0e+00 |
| 512 | 451.9 GFLOPS | 614.3 GFLOPS | 555.7 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 1.4e-15 |
| 1,024 | 482.1 GFLOPS | 774.3 GFLOPS | 656.7 GFLOPS | jax-ynn 2.5e-15, jax-xla-dot 2.7e-15 |
| 4,096 | 420.5 GFLOPS | 602.4 GFLOPS | 704.3 GFLOPS | jax-ynn 6.3e-15, jax-xla-dot 6.1e-15 |

Small products in a recurrence, as QuantEcon/project-benchmarks-jax#4 timed them, µs per product (NumPy's includes Python's dispatch; its bare `np.dot` call in brackets):

| n | NumPy | JAX, YNNPACK | JAX, XLA's own dot |
|---:|---:|---:|---:|
| 8 | 0.800 (0.223) | 0.176 | 0.166 |
| 64 | 4.609 (2.273) | 8.980 | 11.395 |

## The cliff (QuantEcon/project-benchmarks-jax#7), float64, GB/s of the m x m array (warm median)

| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 1.91 | 199.1 (ynn) | 79.5 (xla) | 126.1 (xla) | 246.2 (ynn) |
| 18,000 | 2.41 | 37.7 (ynn) | 79.5 (xla) | 126.6 (xla) | 77.7 (ynn) |

## One value-function kernel, MLX on its CPU device against JAX, float64 (warm median)

| Kernel | n | JAX | MLX | MLX's difference from NumPy |
|---|---:|---:|---:|---|
| stored Bellman step | 2,000 | 0.17 ms | 1.95 ms | 0.0e+00 |
| stored Bellman step | 8,000 | 2.32 ms | 31.86 ms | 0.0e+00 |
| stored Bellman step | 14,000 | 7.71 ms | 105.82 ms | 0.0e+00 |
| P @ V | 1,000 | 2.66 ms | 4.14 ms | 0.0e+00 |
| P @ V | 4,000 | 215.33 ms | 302.57 ms | 0.0e+00 |
