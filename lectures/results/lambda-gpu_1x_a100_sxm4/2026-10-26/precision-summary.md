## lambda-gpu_1x_a100_sxm4 (cuda12)

AMD EPYC 7J13 64-Core Processor, 30 threads, 216 GiB; GPU: NVIDIA A100-SXM4-40GB, 40960 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | cpu | 0.341 | 0.429 | 1.3x | 98.3 | 78.3 | GFLOPS | 0.141 | 5.3e-07 |
| matmul | 512 | cpu | 0.476 | 0.69 | 1.4x | 563 | 389 | GFLOPS | 0.144 | 7.8e-07 |
| matmul | 1024 | cpu | 1.53 | 3.08 | 2.0x | 1.4e+03 | 697 | GFLOPS | 0.146 | 1.3e-06 |
| matmul | 2048 | cpu | 34.3 | 43.1 | 1.3x | 500 | 399 | GFLOPS | 0.186 | 6.9e-07 |
| matmul | 4096 | cpu | 108 | 192 | 1.8x | 1.27e+03 | 718 | GFLOPS | 0.353 | 5.6e-07 |
| matmul | 8192 | cpu | 618 | 1.43e+03 | 2.3x | 1.78e+03 | 770 | GFLOPS | 1.53 | 4.7e-07 |
| matmul | 256 | gpu | 0.165 | 0.163 | 1.0x | 204 | 205 | GFLOPS | 0.856 | 5.5e-05 |
| matmul | 512 | gpu | 0.149 | 0.196 | 1.3x | 1.81e+03 | 1.37e+03 | GFLOPS | 0.859 | 3.9e-05 |
| matmul | 1024 | gpu | 0.237 | 0.338 | 1.4x | 9.06e+03 | 6.36e+03 | GFLOPS | 0.87 | 3.8e-05 |
| matmul | 2048 | gpu | 0.388 | 1.41 | 3.6x | 4.43e+04 | 1.22e+04 | GFLOPS | 0.892 | 4.0e-05 |
| matmul | 4096 | gpu | 1.53 | 7.64 | 5.0x | 9.01e+04 | 1.8e+04 | GFLOPS | 1.4 | 6.1e-05 |
| matmul | 8192 | gpu | 9.21 | 57.4 | 6.2x | 1.19e+05 | 1.92e+04 | GFLOPS | 4.88 | 1.1e-04 |
| bellman_stored | 1000 | cpu | 0.31 | 0.385 | 1.2x | 12.9 | 20.8 | GB/s | 0.158 | 4.3e-07 |
| bellman_stored | 2000 | cpu | 0.382 | 0.459 | 1.2x | 41.9 | 69.7 | GB/s | 0.161 | 4.3e-07 |
| bellman_stored | 5000 | cpu | 1.46 | 2.89 | 2.0x | 68.7 | 69.3 | GB/s | 0.153 | 4.3e-07 |
| bellman_stored | 10000 | cpu | 6.02 | 7.43 | 1.2x | 66.5 | 108 | GB/s | 0.153 | 4.3e-07 |
| bellman_stored | 20000 | cpu | 13.6 | 900 | 66.1x | 118 | 3.56 | GB/s | 1.04 | 4.3e-07 |
| bellman_stored | 40000 | cpu | 1.84e+03 | 1.56e+03 | 0.8x | 3.47 | 8.21 | GB/s | 1.69 | 4.3e-07 |
| bellman_stored | 60000 | cpu | 2.33e+03 | 2.68e+03 | 1.1x | 6.17 | 10.7 | GB/s | 2.88 | 4.3e-07 |
| bellman_stored | 1000 | gpu | 0.158 | 0.127 | 0.8x | 25.3 | 62.9 | GB/s | 0.681 | 4.3e-07 |
| bellman_stored | 2000 | gpu | 0.165 | 0.157 | 1.0x | 97.1 | 204 | GB/s | 0.748 | 4.3e-07 |
| bellman_stored | 5000 | gpu | 0.237 | 0.304 | 1.3x | 423 | 658 | GB/s | 0.599 | 4.3e-07 |
| bellman_stored | 10000 | gpu | 0.446 | 0.751 | 1.7x | 896 | 1.07e+03 | GB/s | 0.649 | 4.3e-07 |
| bellman_stored | 20000 | gpu | 1.3 | 2.51 | 1.9x | 1.23e+03 | 1.27e+03 | GB/s | 0.92 | 4.3e-07 |
| bellman_stored | 40000 | gpu | 4.78 | 9.28 | 1.9x | 1.34e+03 | 1.38e+03 | GB/s | 1.85 | 4.3e-07 |
| bellman_stored | 60000 | gpu | 10.3 | error |  | 1.4e+03 |  | GB/s |  |  |
| bellman_recompute | 1000 | cpu | 0.83 | 3.92 | 4.7x | 1.2 | 0.255 | Gcells/s | 0.241 | 4.3e-07 |
| bellman_recompute | 2000 | cpu | 2.99 | 13.6 | 4.5x | 1.34 | 0.295 | Gcells/s | 0.259 | 4.3e-07 |
| bellman_recompute | 5000 | cpu | 11.5 | 69.2 | 6.0x | 2.17 | 0.361 | Gcells/s | 0.32 | 4.3e-07 |
| bellman_recompute | 10000 | cpu | 39.4 | 290 | 7.4x | 2.54 | 0.344 | Gcells/s | 0.486 | 4.3e-07 |
| bellman_recompute | 20000 | cpu | 162 | 2.52e+03 | 15.6x | 2.47 | 0.159 | Gcells/s | 2.77 | 4.3e-07 |
| bellman_recompute | 40000 | cpu | 4.34e+03 | 6.56e+03 | 1.5x | 0.368 | 0.244 | Gcells/s | 6.71 | 4.3e-07 |
| bellman_recompute | 60000 | cpu | 6.86e+03 | 1.2e+04 | 1.7x | 0.524 | 0.301 | Gcells/s | 12.2 | 4.3e-07 |
| bellman_recompute | 1000 | gpu | 0.143 | 0.185 | 1.3x | 7.01 | 5.39 | Gcells/s | 1.13 | 4.3e-07 |
| bellman_recompute | 2000 | gpu | 0.193 | 0.291 | 1.5x | 20.7 | 13.8 | Gcells/s | 0.921 | 4.3e-07 |
| bellman_recompute | 5000 | gpu | 0.412 | 0.658 | 1.6x | 60.7 | 38 | Gcells/s | 1.01 | 4.3e-07 |
| bellman_recompute | 10000 | gpu | 0.794 | 1.74 | 2.2x | 126 | 57.4 | Gcells/s | 3.01 | 4.3e-07 |
| bellman_recompute | 20000 | gpu | 2.66 | 6.19 | 2.3x | 151 | 64.6 | Gcells/s | 23.6 | 4.3e-07 |
| bellman_recompute | 40000 | gpu | 8.04 | 23.7 | 3.0x | 199 | 67.4 | Gcells/s | 459 | 4.3e-07 |
| bellman_recompute | 60000 | gpu | 13.7 | 36.5 | 2.7x | 262 | 98.6 | Gcells/s | 444 | 4.3e-07 |
| kalman | 1000 | cpu | 6.8 | 6.47 | 1.0x | 147 | 154 | ksteps/s | 0.497 | 1.2e-06 |
| kalman | 10000 | cpu | 81.3 | 76.7 | 0.9x | 123 | 130 | ksteps/s | 0.559 | 6.5e-06 |
| kalman | 100000 | cpu | 797 | 752 | 0.9x | 125 | 133 | ksteps/s | 1.24 | 5.7e-06 |
| kalman | 1000 | gpu | 93.5 | 118 | 1.3x | 10.7 | 8.49 | ksteps/s | 0.898 | 3.5e-02 |
| kalman | 10000 | gpu | 942 | 1.17e+03 | 1.2x | 10.6 | 8.51 | ksteps/s | 1.97 | 8.9e-03 |
| kalman | 100000 | gpu | 9e+03 | 1.14e+04 | 1.3x | 11.1 | 8.75 | ksteps/s | 12.5 | 3.2e-03 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | 256 | not up to 8192 |
| matmul | float64 | gpu | 256 | not up to 8192 |
| bellman_stored | float32 | gpu | 1000 | 40000 |
| bellman_stored | float64 | gpu | 1000 | 20000 |
| bellman_recompute | float32 | gpu | 1000 | not up to 60000 |
| bellman_recompute | float64 | gpu | 1000 | not up to 60000 |
| kalman | float32 | gpu | not up to 100000 | not up to 100000 |
| kalman | float64 | gpu | not up to 100000 | not up to 100000 |

## lambda-gpu_1x_a100_sxm4 (cuda12-highest)

AMD EPYC 7J13 64-Core Processor, 30 threads, 216 GiB; GPU: NVIDIA A100-SXM4-40GB, 40960 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.155 | 0.158 | 1.0x | 216 | 212 | GFLOPS | 0.858 | 5.3e-07 |
| matmul | 512 | gpu | 0.194 | 0.193 | 1.0x | 1.38e+03 | 1.39e+03 | GFLOPS | 0.863 | 7.8e-07 |
| matmul | 1024 | gpu | 0.349 | 0.332 | 1.0x | 6.16e+03 | 6.47e+03 | GFLOPS | 0.856 | 2.5e-07 |
| matmul | 2048 | gpu | 1.42 | 1.4 | 1.0x | 1.21e+04 | 1.23e+04 | GFLOPS | 0.89 | 1.7e-06 |
| matmul | 4096 | gpu | 7.59 | 7.65 | 1.0x | 1.81e+04 | 1.8e+04 | GFLOPS | 1.39 | 2.7e-06 |
| matmul | 8192 | gpu | 57.8 | 57.5 | 1.0x | 1.9e+04 | 1.91e+04 | GFLOPS | 4.9 | 3.4e-06 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

