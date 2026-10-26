## lambda-gpu_1x_a10 (cuda12)

Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz, 30 threads, 222 GiB; GPU: NVIDIA A10, 23028 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | cpu | 0.245 | 0.276 | 1.1x | 137 | 122 | GFLOPS | 0.0933 | 5.3e-07 |
| matmul | 512 | cpu | 0.269 | 0.424 | 1.6x | 998 | 632 | GFLOPS | 0.0947 | 7.8e-07 |
| matmul | 1024 | cpu | 0.809 | 1.36 | 1.7x | 2.66e+03 | 1.58e+03 | GFLOPS | 0.0971 | 5.6e-07 |
| matmul | 2048 | cpu | 13.1 | 20.3 | 1.6x | 1.31e+03 | 844 | GFLOPS | 0.114 | 5.2e-07 |
| matmul | 4096 | cpu | 62.4 | 129 | 2.1x | 2.2e+03 | 1.07e+03 | GFLOPS | 0.222 | 3.6e-07 |
| matmul | 8192 | cpu | 411 | 983 | 2.4x | 2.67e+03 | 1.12e+03 | GFLOPS | 1.05 | 2.4e-07 |
| matmul | 256 | gpu | 0.0887 | 0.173 | 2.0x | 378 | 194 | GFLOPS | 2 | 5.5e-05 |
| matmul | 512 | gpu | 0.0934 | 0.864 | 9.2x | 2.87e+03 | 311 | GFLOPS | 1.98 | 3.9e-05 |
| matmul | 1024 | gpu | 0.142 | 5.01 | 35.3x | 1.51e+04 | 428 | GFLOPS | 2.03 | 3.8e-05 |
| matmul | 2048 | gpu | 0.649 | 36 | 55.5x | 2.65e+04 | 477 | GFLOPS | 2.34 | 4.0e-05 |
| matmul | 4096 | gpu | 3.45 | 284 | 82.3x | 3.98e+04 | 484 | GFLOPS | 5.5 | 6.1e-05 |
| matmul | 8192 | gpu | 26.7 | 2.26e+03 | 84.7x | 4.12e+04 | 486 | GFLOPS | 29.8 | 1.1e-04 |
| bellman_stored | 1000 | cpu | 0.226 | 0.261 | 1.2x | 17.7 | 30.6 | GB/s | 0.107 | 4.3e-07 |
| bellman_stored | 2000 | cpu | 0.319 | 0.299 | 0.9x | 50.2 | 107 | GB/s | 0.108 | 4.3e-07 |
| bellman_stored | 5000 | cpu | 0.691 | 1.41 | 2.0x | 145 | 141 | GB/s | 0.101 | 4.3e-07 |
| bellman_stored | 10000 | cpu | 2.59 | 5.08 | 2.0x | 154 | 157 | GB/s | 0.102 | 4.3e-07 |
| bellman_stored | 20000 | cpu | 10.6 | 576 | 54.1x | 150 | 5.56 | GB/s | 0.675 | 4.3e-07 |
| bellman_stored | 40000 | cpu | 1.16e+03 | 839 | 0.7x | 5.5 | 15.3 | GB/s | 0.886 | 4.3e-07 |
| bellman_stored | 60000 | cpu | 1.33e+03 | 1.38e+03 | 1.0x | 10.8 | 20.9 | GB/s | 1.47 | 4.3e-07 |
| bellman_stored | 1000 | gpu | 0.0863 | 0.144 | 1.7x | 46.4 | 55.4 | GB/s | 0.482 | 4.3e-07 |
| bellman_stored | 2000 | gpu | 0.139 | 0.233 | 1.7x | 115 | 137 | GB/s | 0.528 | 4.3e-07 |
| bellman_stored | 5000 | gpu | 0.308 | 0.989 | 3.2x | 325 | 202 | GB/s | 0.42 | 4.3e-07 |
| bellman_stored | 10000 | gpu | 0.938 | 2.4 | 2.6x | 427 | 333 | GB/s | 0.478 | 4.3e-07 |
| bellman_stored | 20000 | gpu | 3.44 | 13.2 | 3.9x | 466 | 242 | GB/s | 1.44 | 4.3e-07 |
| bellman_stored | 40000 | gpu | 13.4 | error |  | 476 |  | GB/s |  |  |
| bellman_stored | 60000 | gpu | error | skipped |  |  |  | GB/s |  |  |
| bellman_recompute | 1000 | cpu | 0.775 | 2.95 | 3.8x | 1.29 | 0.339 | Gcells/s | 0.152 | 4.3e-07 |
| bellman_recompute | 2000 | cpu | 2.41 | 6.37 | 2.6x | 1.66 | 0.628 | Gcells/s | 0.173 | 4.3e-07 |
| bellman_recompute | 5000 | cpu | 6.43 | 31.8 | 5.0x | 3.89 | 0.785 | Gcells/s | 0.201 | 4.3e-07 |
| bellman_recompute | 10000 | cpu | 21.6 | 121 | 5.6x | 4.62 | 0.83 | Gcells/s | 0.297 | 4.3e-07 |
| bellman_recompute | 20000 | cpu | 67.7 | 1.66e+03 | 24.6x | 5.91 | 0.24 | Gcells/s | 1.82 | 4.3e-07 |
| bellman_recompute | 40000 | cpu | 2.78e+03 | 4.67e+03 | 1.7x | 0.576 | 0.342 | Gcells/s | 4.75 | 4.3e-07 |
| bellman_recompute | 60000 | cpu | 4.46e+03 | 8.84e+03 | 2.0x | 0.806 | 0.407 | Gcells/s | 8.95 | 4.3e-07 |
| bellman_recompute | 1000 | gpu | 0.0964 | 0.328 | 3.4x | 10.4 | 3.05 | Gcells/s | 0.649 | 4.3e-07 |
| bellman_recompute | 2000 | gpu | 0.157 | 1.14 | 7.2x | 25.5 | 3.52 | Gcells/s | 0.673 | 4.3e-07 |
| bellman_recompute | 5000 | gpu | 0.298 | 5.34 | 17.9x | 83.9 | 4.68 | Gcells/s | 1.08 | 4.3e-07 |
| bellman_recompute | 10000 | gpu | timeout | 20.5 |  |  | 4.87 | Gcells/s | 2.18 |  |
| bellman_recompute | 20000 | gpu | 2.43 | 73.8 | 30.3x | 164 | 5.42 | Gcells/s | 417 | 4.3e-07 |
| bellman_recompute | 40000 | gpu | 7.62 | 318 | 41.7x | 210 | 5.04 | Gcells/s | 416 | 4.3e-07 |
| bellman_recompute | 60000 | gpu | 17.8 | 650 | 36.5x | 202 | 5.54 | Gcells/s | 421 | 4.3e-07 |
| kalman | 1000 | cpu | 7.28 | 6.4 | 0.9x | 137 | 156 | ksteps/s | 0.362 | 1.9e-05 |
| kalman | 10000 | cpu | 82.3 | 75.4 | 0.9x | 122 | 133 | ksteps/s | 0.434 | 1.4e-06 |
| kalman | 100000 | cpu | 830 | 743 | 0.9x | 120 | 135 | ksteps/s | 1.08 | 5.7e-06 |
| kalman | 1000 | gpu | 82.4 | 108 | 1.3x | 12.1 | 9.28 | ksteps/s | 0.709 | 3.5e-02 |
| kalman | 10000 | gpu | 799 | 1.06e+03 | 1.3x | 12.5 | 9.47 | ksteps/s | 1.67 | 8.9e-03 |
| kalman | 100000 | gpu | 8e+03 | 1.06e+04 | 1.3x | 12.5 | 9.43 | ksteps/s | 11.5 | 3.2e-03 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | 256 | not up to 8192 |
| matmul | float64 | gpu | 256 | not up to 8192 |
| bellman_stored | float32 | gpu | 1000 | 40000 |
| bellman_stored | float64 | gpu | 1000 | not up to 20000 |
| bellman_recompute | float32 | gpu | 1000 | not up to 60000 |
| bellman_recompute | float64 | gpu | 1000 | not up to 60000 |
| kalman | float32 | gpu | not up to 100000 | not up to 100000 |
| kalman | float64 | gpu | not up to 100000 | not up to 100000 |

## lambda-gpu_1x_a10 (cuda12-highest)

Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz, 30 threads, 222 GiB; GPU: NVIDIA A10, 23028 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.0845 | 0.188 | 2.2x | 397 | 178 | GFLOPS | 1.96 | 5.3e-07 |
| matmul | 512 | gpu | 0.111 | 0.919 | 8.2x | 2.41e+03 | 292 | GFLOPS | 1.99 | 7.8e-07 |
| matmul | 1024 | gpu | 0.238 | 4.92 | 20.6x | 9.01e+03 | 437 | GFLOPS | 2.01 | 1.3e-06 |
| matmul | 2048 | gpu | 1.24 | 36 | 28.9x | 1.38e+04 | 477 | GFLOPS | 2.35 | 1.7e-06 |
| matmul | 4096 | gpu | 8.16 | 284 | 34.8x | 1.68e+04 | 484 | GFLOPS | 5.5 | 2.7e-06 |
| matmul | 8192 | gpu | 72.7 | 2.26e+03 | 31.1x | 1.51e+04 | 486 | GFLOPS | 29.8 | 1.1e-06 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

