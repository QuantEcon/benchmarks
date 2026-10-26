## lambda-gpu_1x_h100_sxm5 (cuda12)

Intel(R) Xeon(R) Platinum 8480+, 26 threads, 221 GiB; GPU: NVIDIA H100 80GB HBM3, 81559 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | cpu | 0.258 | 0.452 | 1.8x | 130 | 74.2 | GFLOPS | 0.0982 | 5.3e-07 |
| matmul | 512 | cpu | 0.454 | 0.706 | 1.6x | 591 | 380 | GFLOPS | 0.0988 | 7.8e-07 |
| matmul | 1024 | cpu | 1.78 | 3.19 | 1.8x | 1.2e+03 | 674 | GFLOPS | 0.101 | 5.6e-07 |
| matmul | 2048 | cpu | 19 | 37.6 | 2.0x | 903 | 457 | GFLOPS | 0.136 | 5.2e-07 |
| matmul | 4096 | cpu | 122 | 208 | 1.7x | 1.13e+03 | 662 | GFLOPS | 0.304 | 3.6e-07 |
| matmul | 8192 | cpu | 752 | 1.61e+03 | 2.1x | 1.46e+03 | 684 | GFLOPS | 1.7 | 2.4e-07 |
| matmul | 256 | gpu | 0.0993 | 0.109 | 1.1x | 338 | 307 | GFLOPS | 0.805 | 5.3e-05 |
| matmul | 512 | gpu | 0.107 | 0.144 | 1.3x | 2.51e+03 | 1.86e+03 | GFLOPS | 0.954 | 3.9e-05 |
| matmul | 1024 | gpu | 0.16 | 0.168 | 1.1x | 1.34e+04 | 1.28e+04 | GFLOPS | 0.96 | 3.5e-05 |
| matmul | 2048 | gpu | 0.196 | 0.438 | 2.2x | 8.77e+04 | 3.92e+04 | GFLOPS | 0.969 | 2.9e-05 |
| matmul | 4096 | gpu | 0.489 | 2.31 | 4.7x | 2.81e+05 | 5.95e+04 | GFLOPS | 1.06 | 3.8e-05 |
| matmul | 8192 | gpu | 2.76 | 17.2 | 6.2x | 3.99e+05 | 6.39e+04 | GFLOPS | 1.63 | 6.4e-05 |
| bellman_stored | 1000 | cpu | 0.259 | 0.319 | 1.2x | 15.5 | 25.1 | GB/s | 0.113 | 4.3e-07 |
| bellman_stored | 2000 | cpu | 0.335 | 0.432 | 1.3x | 47.7 | 74 | GB/s | 0.114 | 4.3e-07 |
| bellman_stored | 5000 | cpu | 0.808 | 1.56 | 1.9x | 124 | 128 | GB/s | 0.103 | 4.3e-07 |
| bellman_stored | 10000 | cpu | 3.11 | 6.27 | 2.0x | 129 | 128 | GB/s | 0.107 | 4.3e-07 |
| bellman_stored | 20000 | cpu | 12.3 | 601 | 49.0x | 130 | 5.32 | GB/s | 0.721 | 4.3e-07 |
| bellman_stored | 40000 | cpu | 1.24e+03 | 813 | 0.7x | 5.17 | 15.7 | GB/s | 0.916 | 4.3e-07 |
| bellman_stored | 60000 | cpu | 1.38e+03 | 1.41e+03 | 1.0x | 10.4 | 20.4 | GB/s | 1.56 | 4.3e-07 |
| bellman_stored | 1000 | gpu | 0.11 | 0.12 | 1.1x | 36.5 | 66.5 | GB/s | 0.662 | 4.3e-07 |
| bellman_stored | 2000 | gpu | 0.109 | 0.136 | 1.3x | 147 | 235 | GB/s | 0.765 | 4.3e-07 |
| bellman_stored | 5000 | gpu | 0.15 | 0.188 | 1.3x | 667 | 1.07e+03 | GB/s | 0.564 | 4.3e-07 |
| bellman_stored | 10000 | gpu | 0.256 | 0.42 | 1.6x | 1.56e+03 | 1.9e+03 | GB/s | 0.536 | 4.3e-07 |
| bellman_stored | 20000 | gpu | 0.67 | 1.19 | 1.8x | 2.39e+03 | 2.7e+03 | GB/s | 0.737 | 4.3e-07 |
| bellman_stored | 40000 | gpu | 2.21 | 4.27 | 1.9x | 2.9e+03 | 3e+03 | GB/s | 1.2 | 4.3e-07 |
| bellman_stored | 60000 | gpu | 4.76 | 9.39 | 2.0x | 3.02e+03 | 3.07e+03 | GB/s | 1.52 | 4.3e-07 |
| bellman_recompute | 1000 | cpu | 0.685 | 2.45 | 3.6x | 1.46 | 0.407 | Gcells/s | 0.158 | 4.3e-07 |
| bellman_recompute | 2000 | cpu | 2.36 | 7.99 | 3.4x | 1.69 | 0.501 | Gcells/s | 0.183 | 4.3e-07 |
| bellman_recompute | 5000 | cpu | 8.14 | 47.7 | 5.9x | 3.07 | 0.524 | Gcells/s | 0.218 | 4.3e-07 |
| bellman_recompute | 10000 | cpu | 28.6 | 176 | 6.2x | 3.5 | 0.567 | Gcells/s | 0.355 | 4.3e-07 |
| bellman_recompute | 20000 | cpu | 108 | 1.68e+03 | 15.6x | 3.71 | 0.238 | Gcells/s | 1.85 | 4.3e-07 |
| bellman_recompute | 40000 | cpu | 2.77e+03 | 4.64e+03 | 1.7x | 0.577 | 0.345 | Gcells/s | 4.78 | 4.3e-07 |
| bellman_recompute | 60000 | cpu | 4.63e+03 | 8.7e+03 | 1.9x | 0.777 | 0.414 | Gcells/s | 8.79 | 4.3e-07 |
| bellman_recompute | 1000 | gpu | 0.105 | 0.111 | 1.1x | 9.55 | 9 | Gcells/s | 0.758 | 4.3e-07 |
| bellman_recompute | 2000 | gpu | 0.169 | 0.166 | 1.0x | 23.7 | 24 | Gcells/s | 0.995 | 4.3e-07 |
| bellman_recompute | 5000 | gpu | 0.259 | 0.328 | 1.3x | 96.7 | 76.2 | Gcells/s | 0.91 | 4.3e-07 |
| bellman_recompute | 10000 | gpu | 0.414 | 0.731 | 1.8x | 242 | 137 | Gcells/s | 1.3 | 4.3e-07 |
| bellman_recompute | 20000 | gpu | 1.14 | 2.31 | 2.0x | 351 | 173 | Gcells/s | 2.33 | 4.3e-07 |
| bellman_recompute | 40000 | gpu | 3.42 | 8.52 | 2.5x | 468 | 188 | Gcells/s | 158 | 4.3e-07 |
| bellman_recompute | 60000 | gpu | 5.96 | 16.6 | 2.8x | 604 | 216 | Gcells/s | 155 | 4.3e-07 |
| kalman | 1000 | cpu | 6.02 | 5.38 | 0.9x | 166 | 186 | ksteps/s | 0.387 | 1.9e-05 |
| kalman | 10000 | cpu | 70.4 | 63.8 | 0.9x | 142 | 157 | ksteps/s | 0.447 | 1.7e-06 |
| kalman | 100000 | cpu | 701 | 634 | 0.9x | 143 | 158 | ksteps/s | 1.01 | 5.7e-06 |
| kalman | 1000 | gpu | 90.1 | 93 | 1.0x | 11.1 | 10.8 | ksteps/s | 0.829 | 5.4e-02 |
| kalman | 10000 | gpu | 879 | 928 | 1.1x | 11.4 | 10.8 | ksteps/s | 1.69 | 1.9e-02 |
| kalman | 100000 | gpu | 8.69e+03 | 8.99e+03 | 1.0x | 11.5 | 11.1 | ksteps/s | 9.87 | 2.2e-03 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | 256 | not up to 8192 |
| matmul | float64 | gpu | 256 | 8192 |
| bellman_stored | float32 | gpu | 1000 | 40000 |
| bellman_stored | float64 | gpu | 1000 | 60000 |
| bellman_recompute | float32 | gpu | 1000 | 60000 |
| bellman_recompute | float64 | gpu | 1000 | not up to 60000 |
| kalman | float32 | gpu | not up to 100000 | not up to 100000 |
| kalman | float64 | gpu | not up to 100000 | not up to 100000 |

## lambda-gpu_1x_h100_sxm5 (cuda12-highest)

Intel(R) Xeon(R) Platinum 8480+, 26 threads, 221 GiB; GPU: NVIDIA H100 80GB HBM3, 81559 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.118 | 0.106 | 0.9x | 284 | 317 | GFLOPS | 0.818 | 2.0e-07 |
| matmul | 512 | gpu | 0.108 | 0.136 | 1.3x | 2.48e+03 | 1.97e+03 | GFLOPS | 0.952 | 2.3e-07 |
| matmul | 1024 | gpu | 0.2 | 0.169 | 0.8x | 1.07e+04 | 1.27e+04 | GFLOPS | 0.952 | 2.3e-07 |
| matmul | 2048 | gpu | 0.482 | 0.425 | 0.9x | 3.57e+04 | 4.04e+04 | GFLOPS | 1.02 | 1.7e-06 |
| matmul | 4096 | gpu | 2.81 | 2.31 | 0.8x | 4.9e+04 | 5.95e+04 | GFLOPS | 1.06 | 2.7e-06 |
| matmul | 8192 | gpu | 21.4 | 17.3 | 0.8x | 5.14e+04 | 6.34e+04 | GFLOPS | 1.64 | 1.1e-06 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

