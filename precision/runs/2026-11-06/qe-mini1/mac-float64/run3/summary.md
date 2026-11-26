## Matrix products, float64

| n | NumPy (Accelerate) | JAX, YNNPACK | JAX, XLA's own dot | Largest relative difference from NumPy |
|---:|---:|---:|---:|---|
| 8 | 3.5 GFLOPS | 0.3 GFLOPS | 0.3 GFLOPS | jax-ynn 1.7e-16, jax-xla-dot 1.7e-16 |
| 64 | 218.9 GFLOPS | 27.2 GFLOPS | 23.1 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 0.0e+00 |
| 512 | 493.5 GFLOPS | 401.5 GFLOPS | 349.4 GFLOPS | jax-ynn 0.0e+00, jax-xla-dot 1.4e-15 |
| 1,024 | 556.6 GFLOPS | 427.5 GFLOPS | 409.3 GFLOPS | jax-ynn 2.5e-15, jax-xla-dot 2.7e-15 |
| 4,096 | 542.9 GFLOPS | 360.8 GFLOPS | 461.7 GFLOPS | jax-ynn 6.3e-15, jax-xla-dot 6.1e-15 |

Small products in a recurrence, as QuantEcon/project-benchmarks-jax#4 timed them, µs per product (NumPy's includes Python's dispatch; its bare `np.dot` call in brackets):

| n | NumPy | JAX, YNNPACK | JAX, XLA's own dot |
|---:|---:|---:|---:|
| 8 | 0.695 (0.205) | 0.154 | 0.145 |
| 64 | 4.117 (2.439) | 8.358 | 10.337 |

## The cliff (QuantEcon/project-benchmarks-jax#7), float64, GB/s of the m x m array (warm median)

| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 1.91 | 130.7 (ynn) | 45.9 (xla) | 59.4 (xla) | 158.7 (ynn) |
| 18,000 | 2.41 | 35.2 (ynn) | 46.0 (xla) | 60.5 (xla) | 80.9 (ynn) |

## One value-function kernel, MLX on its CPU device against JAX, float64 (warm median)

| Kernel | n | JAX | MLX | MLX's difference from NumPy |
|---|---:|---:|---:|---|
| stored Bellman step | 2,000 | 0.21 ms | 2.21 ms | 0.0e+00 |
| stored Bellman step | 8,000 | 3.54 ms | 36.46 ms | 0.0e+00 |
| stored Bellman step | 14,000 | 12.05 ms | 112.11 ms | 0.0e+00 |
| P @ V | 1,000 | 4.90 ms | 3.51 ms | 0.0e+00 |
| P @ V | 4,000 | 350.71 ms | 229.55 ms | 0.0e+00 |
