## runpod-rtx5090

AMD Ryzen 9 9950X 16-Core Processor, 32 threads, 123 GiB; GPU: NVIDIA GeForce RTX 5090, 32607 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.0828 | 0.0861 | 1.0x | 405 | 390 | GFLOPS | 1.5 | 5.3e-05 |
| matmul | 512 | gpu | 0.0979 | 0.21 | 2.1x | 2.74e+03 | 1.28e+03 | GFLOPS | 2.63 | 3.9e-05 |
| matmul | 1024 | gpu | 0.0758 | 1.13 | 14.9x | 2.83e+04 | 1.9e+03 | GFLOPS | 3.2 | 3.5e-05 |
| matmul | 2048 | gpu | 0.223 | 8.5 | 38.2x | 7.72e+04 | 2.02e+03 | GFLOPS | 4.25 | 2.9e-05 |
| matmul | 4096 | gpu | 1.29 | 68.4 | 53.1x | 1.07e+05 | 2.01e+03 | GFLOPS | 9.28 | 2.1e-05 |
| matmul | 8192 | gpu | 8.82 | 535 | 60.7x | 1.25e+05 | 2.05e+03 | GFLOPS | 55.5 | 6.4e-05 |
| bellman_stored | 1000 | gpu | 0.119 | 0.0473 | 0.4x | 33.7 | 169 | GB/s | 0.313 | 4.3e-07 |
| bellman_stored | 2000 | gpu | 0.0456 | 0.102 | 2.2x | 351 | 314 | GB/s | 0.364 | 4.3e-07 |
| bellman_stored | 5000 | gpu | 0.0566 | 0.21 | 3.7x | 1.77e+03 | 954 | GB/s | 0.327 | 4.3e-07 |
| bellman_stored | 10000 | gpu | 0.326 | 0.754 | 2.3x | 1.23e+03 | 1.06e+03 | GB/s | 0.323 | 4.3e-07 |
| bellman_stored | 20000 | gpu | 1.03 | 3.06 | 3.0x | 1.56e+03 | 1.05e+03 | GB/s | 0.477 | 4.3e-07 |
| bellman_stored | 40000 | gpu | 3.93 | 12.1 | 3.1x | 1.63e+03 | 1.06e+03 | GB/s | 0.707 | 4.3e-07 |
| bellman_stored | 60000 | gpu | 8.72 | skipped |  | 1.65e+03 |  | GB/s |  |  |
| bellman_recompute | 1000 | gpu | 0.0983 | 0.16 | 1.6x | 10.2 | 6.27 | Gcells/s | 0.438 | 4.3e-07 |
| bellman_recompute | 2000 | gpu | 0.122 | 0.306 | 2.5x | 32.8 | 13.1 | Gcells/s | 0.585 | 4.3e-07 |
| bellman_recompute | 5000 | gpu | 0.141 | 1.56 | 11.0x | 177 | 16.1 | Gcells/s | 0.549 | 4.3e-07 |
| bellman_recompute | 10000 | gpu | 0.277 | 5.61 | 20.2x | 361 | 17.8 | Gcells/s | 0.761 | 4.3e-07 |
| bellman_recompute | 20000 | gpu | 0.625 | 17.7 | 28.4x | 640 | 22.6 | Gcells/s | 13.5 | 4.3e-07 |
| bellman_recompute | 40000 | gpu | 2.03 | 75.9 | 37.5x | 790 | 21.1 | Gcells/s | 13.7 | 4.3e-07 |
| bellman_recompute | 60000 | gpu | 4.11 | 154 | 37.6x | 877 | 23.3 | Gcells/s | 57.1 | 4.3e-07 |
| kalman | 1000 | gpu | 41.4 | 55.8 | 1.3x | 24.2 | 17.9 | ksteps/s | 0.534 | 5.4e-02 |
| kalman | 10000 | gpu | 413 | 555 | 1.3x | 24.2 | 18 | ksteps/s | 1.03 | 1.9e-02 |
| kalman | 100000 | gpu | 4.12e+03 | 5.54e+03 | 1.3x | 24.3 | 18 | ksteps/s | 6.11 | 2.2e-03 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |
| bellman_stored | float32 | gpu | no comparison | no comparison |
| bellman_stored | float64 | gpu | no comparison | no comparison |
| bellman_recompute | float32 | gpu | no comparison | no comparison |
| bellman_recompute | float64 | gpu | no comparison | no comparison |
| kalman | float32 | gpu | no comparison | no comparison |
| kalman | float64 | gpu | no comparison | no comparison |

## runpod-rtx5090 (highest)

AMD Ryzen 9 9950X 16-Core Processor, 32 threads, 123 GiB; GPU: NVIDIA GeForce RTX 5090, 32607 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.11 | 0.0627 | 0.6x | 306 | 536 | GFLOPS | 1.5 | 1.9e-07 |
| matmul | 512 | gpu | 0.0399 | 0.201 | 5.0x | 6.73e+03 | 1.34e+03 | GFLOPS | 2.73 | 2.3e-07 |
| matmul | 1024 | gpu | 0.0728 | 1.18 | 16.2x | 2.95e+04 | 1.82e+03 | GFLOPS | 3.16 | 1.3e-06 |
| matmul | 2048 | gpu | 0.285 | 8.54 | 30.0x | 6.03e+04 | 2.01e+03 | GFLOPS | 4.24 | 3.6e-07 |
| matmul | 4096 | gpu | 1.89 | 68.5 | 36.3x | 7.28e+04 | 2.01e+03 | GFLOPS | 9.31 | 5.6e-07 |
| matmul | 8192 | gpu | 16.6 | 535 | 32.3x | 6.64e+04 | 2.05e+03 | GFLOPS | 55.7 | 7.9e-07 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

