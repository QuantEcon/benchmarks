## aws-g7e.2xlarge

Intel(R) Xeon(R) Platinum 8559C, 8 threads, 62 GiB; GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | cpu | 0.137 | 0.205 | 1.5x | 245 | 164 | GFLOPS | 0.0801 | 5.3e-07 |
| matmul | 512 | cpu | 0.412 | 0.768 | 1.9x | 651 | 350 | GFLOPS | 0.085 | 7.8e-07 |
| matmul | 1024 | cpu | 2.53 | 4.98 | 2.0x | 850 | 432 | GFLOPS | 0.09 | 5.6e-07 |
| matmul | 2048 | cpu | 19.3 | 44.7 | 2.3x | 892 | 384 | GFLOPS | 0.131 | 5.2e-07 |
| matmul | 4096 | cpu | 166 | 332 | 2.0x | 830 | 414 | GFLOPS | 0.414 | 3.6e-07 |
| matmul | 8192 | cpu | 1.31e+03 | 3.02e+03 | 2.3x | 839 | 364 | GFLOPS | 3.08 | 2.4e-07 |
| matmul | 256 | gpu | 0.0713 | 0.0868 | 1.2x | 471 | 386 | GFLOPS | 3.02 | 5.3e-05 |
| matmul | 512 | gpu | 0.0725 | 0.229 | 3.2x | 3.7e+03 | 1.17e+03 | GFLOPS | 5.35 | 3.9e-05 |
| matmul | 1024 | gpu | 0.114 | 1.3 | 11.5x | 1.89e+04 | 1.65e+03 | GFLOPS | 5.6 | 3.5e-05 |
| matmul | 2048 | gpu | 0.193 | 9.88 | 51.1x | 8.88e+04 | 1.74e+03 | GFLOPS | 7.07 | 2.3e-05 |
| matmul | 4096 | gpu | 0.828 | 78.8 | 95.2x | 1.66e+05 | 1.74e+03 | GFLOPS | 12.6 | 1.7e-05 |
| matmul | 8192 | gpu | 5.92 | 626 | 105.7x | 1.86e+05 | 1.76e+03 | GFLOPS | 66.4 | 6.4e-05 |
| bellman_stored | 1000 | cpu | 0.113 | 0.158 | 1.4x | 35.5 | 50.7 | GB/s | 0.0975 | 4.3e-07 |
| bellman_stored | 2000 | cpu | 0.296 | 0.432 | 1.5x | 54.1 | 74 | GB/s | 0.0922 | 4.3e-07 |
| bellman_stored | 5000 | cpu | 1.43 | 2.61 | 1.8x | 69.7 | 76.8 | GB/s | 0.0883 | 4.3e-07 |
| bellman_stored | 10000 | cpu | 6.38 | 14.5 | 2.3x | 62.7 | 55.3 | GB/s | 0.102 | 4.3e-07 |
| bellman_stored | 20000 | cpu | 28.8 | 526 | 18.3x | 55.6 | 6.08 | GB/s | 0.617 | 4.3e-07 |
| bellman_stored | 40000 | cpu | 1.06e+03 | 1.02e+03 | 1.0x | 6.06 | 12.5 | GB/s | 1.06 | 4.3e-07 |
| bellman_stored | 60000 | cpu | 1.39e+03 | 1.93e+03 | 1.4x | 10.3 | 14.9 | GB/s | 1.98 | 4.3e-07 |
| bellman_stored | 1000 | gpu | 0.0626 | 0.0909 | 1.5x | 63.9 | 88 | GB/s | 0.381 | 4.3e-07 |
| bellman_stored | 2000 | gpu | 0.0457 | 0.114 | 2.5x | 350 | 281 | GB/s | 0.471 | 4.3e-07 |
| bellman_stored | 5000 | gpu | 0.081 | 0.256 | 3.2x | 1.23e+03 | 783 | GB/s | 0.396 | 4.3e-07 |
| bellman_stored | 10000 | gpu | 0.339 | 0.873 | 2.6x | 1.18e+03 | 917 | GB/s | 0.403 | 4.3e-07 |
| bellman_stored | 20000 | gpu | 1.15 | 3.55 | 3.1x | 1.39e+03 | 903 | GB/s | 0.578 | 4.3e-07 |
| bellman_stored | 40000 | gpu | 4.35 | 13.9 | 3.2x | 1.47e+03 | 919 | GB/s | 0.99 | 4.3e-07 |
| bellman_stored | 60000 | gpu | 9.65 | 27.6 | 2.9x | 1.49e+03 | 1.04e+03 | GB/s | 1.33 | 4.3e-07 |
| bellman_recompute | 1000 | cpu | 0.398 | 2.74 | 6.9x | 2.51 | 0.364 | Gcells/s | 0.133 | 4.3e-07 |
| bellman_recompute | 2000 | cpu | 1.41 | 8.19 | 5.8x | 2.84 | 0.488 | Gcells/s | 0.158 | 4.3e-07 |
| bellman_recompute | 5000 | cpu | 7.8 | 52 | 6.7x | 3.2 | 0.481 | Gcells/s | 0.203 | 4.3e-07 |
| bellman_recompute | 10000 | cpu | 28.1 | 214 | 7.6x | 3.56 | 0.467 | Gcells/s | 0.365 | 4.3e-07 |
| bellman_recompute | 20000 | cpu | 124 | 1.53e+03 | 12.4x | 3.23 | 0.261 | Gcells/s | 1.58 | 4.3e-07 |
| bellman_recompute | 40000 | cpu | 2.2e+03 | 4.59e+03 | 2.1x | 0.726 | 0.348 | Gcells/s | 4.73 | 4.3e-07 |
| bellman_recompute | 60000 | cpu | 3.22e+03 | 9.21e+03 | 2.9x | 1.12 | 0.391 | Gcells/s | 9.36 | 4.3e-07 |
| bellman_recompute | 1000 | gpu | 0.0639 | 0.149 | 2.3x | 15.7 | 6.73 | Gcells/s | 0.533 | 4.3e-07 |
| bellman_recompute | 2000 | gpu | 0.15 | 0.41 | 2.7x | 26.6 | 9.76 | Gcells/s | 0.801 | 4.3e-07 |
| bellman_recompute | 5000 | gpu | 0.177 | 1.52 | 8.6x | 142 | 16.5 | Gcells/s | 0.711 | 4.3e-07 |
| bellman_recompute | 10000 | gpu | 0.329 | 5.52 | 16.8x | 304 | 18.1 | Gcells/s | 1.04 | 4.3e-07 |
| bellman_recompute | 20000 | gpu | 0.731 | 20.4 | 27.9x | 547 | 19.6 | Gcells/s | 18.9 | 4.3e-07 |
| bellman_recompute | 40000 | gpu | 2.34 | 88.7 | 37.9x | 684 | 18 | Gcells/s | 18.8 | 4.3e-07 |
| bellman_recompute | 60000 | gpu | 4.45 | 180 | 40.3x | 808 | 20.1 | Gcells/s | 92.5 | 4.3e-07 |
| kalman | 1000 | cpu | 4.24 | 3.81 | 0.9x | 236 | 263 | ksteps/s | 0.329 | 1.9e-05 |
| kalman | 10000 | cpu | 49.4 | 42.7 | 0.9x | 202 | 234 | ksteps/s | 0.371 | 1.7e-06 |
| kalman | 100000 | cpu | 493 | 440 | 0.9x | 203 | 227 | ksteps/s | 0.763 | 5.7e-06 |
| kalman | 1000 | gpu | 68 | 87.8 | 1.3x | 14.7 | 11.4 | ksteps/s | 0.849 | 5.4e-02 |
| kalman | 10000 | gpu | 677 | 868 | 1.3x | 14.8 | 11.5 | ksteps/s | 1.62 | 1.9e-02 |
| kalman | 100000 | gpu | 6.8e+03 | 8.74e+03 | 1.3x | 14.7 | 11.4 | ksteps/s | 9.64 | 2.2e-03 |

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

## aws-g7e.2xlarge (highest)

Intel(R) Xeon(R) Platinum 8559C, 8 threads, 62 GiB; GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.0877 | 0.11 | 1.3x | 383 | 306 | GFLOPS | 2.98 | 1.9e-07 |
| matmul | 512 | gpu | 0.0647 | 0.225 | 3.5x | 4.15e+03 | 1.19e+03 | GFLOPS | 5.19 | 2.3e-07 |
| matmul | 1024 | gpu | 0.119 | 1.3 | 11.0x | 1.81e+04 | 1.65e+03 | GFLOPS | 5.31 | 1.3e-06 |
| matmul | 2048 | gpu | 0.351 | 9.98 | 28.4x | 4.89e+04 | 1.72e+03 | GFLOPS | 6.72 | 1.7e-06 |
| matmul | 4096 | gpu | 2.01 | 78.8 | 39.2x | 6.84e+04 | 1.74e+03 | GFLOPS | 12.4 | 2.7e-06 |
| matmul | 8192 | gpu | 13.3 | 626 | 47.2x | 8.29e+04 | 1.76e+03 | GFLOPS | 66.5 | 1.4e-06 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

