## runpod-rtxpro6000-mig1g24

AMD EPYC 9554 64-Core Processor, 256 threads, 1007 GiB; GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 24192 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.127 | 0.209 | 1.6x | 264 | 161 | GFLOPS | 5.47 | 5.3e-05 |
| matmul | 512 | gpu | 0.133 | 0.845 | 6.3x | 2.01e+03 | 318 | GFLOPS | 12.9 | 3.8e-05 |
| matmul | 1024 | gpu | 0.182 | 5.34 | 29.3x | 1.18e+04 | 402 | GFLOPS | 13.2 | 3.3e-05 |
| matmul | 2048 | gpu | 0.598 | 40.9 | 68.4x | 2.87e+04 | 420 | GFLOPS | 11.6 | 2.9e-05 |
| matmul | 4096 | gpu | 3.51 | 323 | 92.0x | 3.91e+04 | 425 | GFLOPS | 39.9 | 3.8e-05 |
| matmul | 8192 | gpu | 21.4 | 2.57e+03 | 120.4x | 5.15e+04 | 427 | GFLOPS | 262 | 6.4e-05 |
| bellman_stored | 1000 | gpu | 0.188 | 0.192 | 1.0x | 21.3 | 41.6 | GB/s | 0.97 | 4.3e-07 |
| bellman_stored | 2000 | gpu | 0.182 | 0.313 | 1.7x | 88 | 102 | GB/s | 0.844 | 4.3e-07 |
| bellman_stored | 5000 | gpu | 0.417 | 0.98 | 2.4x | 240 | 204 | GB/s | 0.671 | 4.3e-07 |
| bellman_stored | 10000 | gpu | 1.24 | 3.5 | 2.8x | 323 | 228 | GB/s | 0.731 | 4.3e-07 |
| bellman_stored | 20000 | gpu | 4.6 | 14.7 | 3.2x | 348 | 218 | GB/s | 1.21 | 4.3e-07 |
| bellman_stored | 40000 | gpu | 17.4 | error |  | 367 |  | GB/s |  |  |
| bellman_stored | 60000 | gpu | error | skipped |  |  |  | GB/s |  |  |
| bellman_recompute | 1000 | gpu | 0.109 | 0.386 | 3.5x | 9.13 | 2.59 | Gcells/s | 0.987 | 4.3e-07 |
| bellman_recompute | 2000 | gpu | 0.371 | 1.13 | 3.1x | 10.8 | 3.53 | Gcells/s | 1.29 | 4.3e-07 |
| bellman_recompute | 5000 | gpu | 0.379 | 5.68 | 15.0x | 65.9 | 4.4 | Gcells/s | 2.57 | 4.3e-07 |
| bellman_recompute | 10000 | gpu | 1.08 | 27 | 25.0x | 92.5 | 3.7 | Gcells/s | 326 | 4.3e-07 |
| bellman_recompute | 20000 | gpu | 1.99 | 82.7 | 41.6x | 201 | 4.84 | Gcells/s | 82.4 | 4.3e-07 |
| bellman_recompute | 40000 | gpu | 7.26 | 361 | 49.7x | 220 | 4.44 | Gcells/s | 82.9 | 4.3e-07 |
| bellman_recompute | 60000 | gpu | 14.6 | 735 | 50.5x | 247 | 4.9 | Gcells/s | 82.6 | 4.3e-07 |
| kalman | 1000 | gpu | 66.5 | 72.1 | 1.1x | 15 | 13.9 | ksteps/s | 1.27 | 5.4e-02 |
| kalman | 10000 | gpu | 529 | 717 | 1.4x | 18.9 | 13.9 | ksteps/s | 1.96 | 1.9e-02 |
| kalman | 100000 | gpu | 5.23e+03 | 7.17e+03 | 1.4x | 19.1 | 14 | ksteps/s | 8.67 | 2.2e-03 |

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

## runpod-rtxpro6000-mig1g24 (highest)

AMD EPYC 9554 64-Core Processor, 256 threads, 1007 GiB; GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 24192 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.133 | 0.188 | 1.4x | 252 | 179 | GFLOPS | 5.05 | 1.9e-07 |
| matmul | 512 | gpu | 0.122 | 0.807 | 6.6x | 2.2e+03 | 333 | GFLOPS | 13.3 | 3.5e-07 |
| matmul | 1024 | gpu | 0.245 | 5.33 | 21.8x | 8.78e+03 | 403 | GFLOPS | 12.5 | 1.3e-06 |
| matmul | 2048 | gpu | 1.16 | 40.9 | 35.4x | 1.49e+04 | 420 | GFLOPS | 11.9 | 1.7e-06 |
| matmul | 4096 | gpu | 7.43 | 324 | 43.5x | 1.85e+04 | 425 | GFLOPS | 40 | 7.9e-07 |
| matmul | 8192 | gpu | 54.9 | 2.57e+03 | 46.9x | 2e+04 | 427 | GFLOPS | 262 | 3.4e-06 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

