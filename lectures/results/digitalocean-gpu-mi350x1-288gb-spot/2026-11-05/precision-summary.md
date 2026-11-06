## digitalocean-gpu-mi350x1-288gb-spot

AMD EPYC 9575F 64-Core Processor, 24 threads, 252 GiB; GPU: AMD Instinct MI350X VF, 294592 MiB; JAX 0.12.0

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | cpu | 0.133 | 0.129 | 1.0x | 253 | 260 | GFLOPS | 0.052 | 5.3e-07 |
| matmul | 512 | cpu | 0.196 | 0.282 | 1.4x | 1.37e+03 | 952 | GFLOPS | 0.0524 | 7.8e-07 |
| matmul | 1024 | cpu | 0.683 | 1.21 | 1.8x | 3.14e+03 | 1.78e+03 | GFLOPS | 0.0574 | 5.6e-07 |
| matmul | 2048 | cpu | 3.87 | 18.8 | 4.9x | 4.44e+03 | 915 | GFLOPS | 0.0726 | 5.2e-07 |
| matmul | 4096 | cpu | 56.8 | 82.4 | 1.5x | 2.42e+03 | 1.67e+03 | GFLOPS | 0.133 | 3.6e-07 |
| matmul | 8192 | cpu | 289 | 571 | 2.0x | 3.81e+03 | 1.92e+03 | GFLOPS | 0.615 | 2.4e-07 |
| matmul | 256 | gpu | 0.0871 | 0.142 | 1.6x | 385 | 236 | GFLOPS | 1.1 | 5.3e-07 |
| matmul | 512 | gpu | 0.079 | 0.116 | 1.5x | 3.4e+03 | 2.31e+03 | GFLOPS | 1.12 | 7.8e-07 |
| matmul | 1024 | gpu | 0.154 | 0.158 | 1.0x | 1.39e+04 | 1.36e+04 | GFLOPS | 1.12 | 1.1e-06 |
| matmul | 2048 | gpu | 0.308 | 0.381 | 1.2x | 5.57e+04 | 4.51e+04 | GFLOPS | 1.15 | 1.7e-06 |
| matmul | 4096 | gpu | 1.38 | 2.37 | 1.7x | 9.99e+04 | 5.79e+04 | GFLOPS | 1.68 | 2.6e-06 |
| matmul | 8192 | gpu | 7.99 | 15.8 | 2.0x | 1.38e+05 | 6.98e+04 | GFLOPS | 5.85 | 3.4e-06 |
| bellman_stored | 1000 | cpu | 0.117 | 0.129 | 1.1x | 34.1 | 61.8 | GB/s | 0.0616 | 4.3e-07 |
| bellman_stored | 2000 | cpu | 0.125 | 0.166 | 1.3x | 128 | 193 | GB/s | 0.0644 | 4.3e-07 |
| bellman_stored | 5000 | cpu | 0.334 | 0.7 | 2.1x | 300 | 286 | GB/s | 0.0551 | 4.3e-07 |
| bellman_stored | 10000 | cpu | 1.34 | 2.59 | 1.9x | 299 | 309 | GB/s | 0.0599 | 4.3e-07 |
| bellman_stored | 20000 | cpu | 4.62 | 323 | 70.0x | 347 | 9.9 | GB/s | 0.323 | 4.3e-07 |
| bellman_stored | 40000 | cpu | 687 | 472 | 0.7x | 9.31 | 27.1 | GB/s | 0.504 | 4.3e-07 |
| bellman_stored | 60000 | cpu | 952 | 719 | 0.8x | 15.1 | 40.1 | GB/s | 0.784 | 4.3e-07 |
| bellman_stored | 1000 | gpu | 0.0811 | 0.0868 | 1.1x | 49.3 | 92.2 | GB/s | 0.243 | 3.5e-07 |
| bellman_stored | 2000 | gpu | 0.085 | 0.111 | 1.3x | 188 | 289 | GB/s | 0.258 | 3.5e-07 |
| bellman_stored | 5000 | gpu | 0.149 | 0.154 | 1.0x | 670 | 1.3e+03 | GB/s | 0.267 | 3.5e-07 |
| bellman_stored | 10000 | gpu | 0.154 | 0.248 | 1.6x | 2.6e+03 | 3.22e+03 | GB/s | 0.278 | 3.5e-07 |
| bellman_stored | 20000 | gpu | 0.593 | 1.1 | 1.8x | 2.7e+03 | 2.92e+03 | GB/s | 0.306 | 3.5e-07 |
| bellman_stored | 40000 | gpu | 1.62 | 3.4 | 2.1x | 3.95e+03 | 3.77e+03 | GB/s | 0.398 | 3.5e-07 |
| bellman_stored | 60000 | gpu | 3.12 | 6.92 | 2.2x | 4.61e+03 | 4.16e+03 | GB/s | 0.94 | 3.5e-07 |
| bellman_recompute | 1000 | cpu | 0.359 | 1.61 | 4.5x | 2.78 | 0.62 | Gcells/s | 0.0871 | 4.3e-07 |
| bellman_recompute | 2000 | cpu | 0.898 | 6.65 | 7.4x | 4.45 | 0.601 | Gcells/s | 0.108 | 4.3e-07 |
| bellman_recompute | 5000 | cpu | 6.64 | 34.8 | 5.2x | 3.77 | 0.718 | Gcells/s | 0.129 | 4.3e-07 |
| bellman_recompute | 10000 | cpu | 22.7 | 148 | 6.5x | 4.41 | 0.675 | Gcells/s | 0.255 | 4.3e-07 |
| bellman_recompute | 20000 | cpu | 91.8 | 913 | 9.9x | 4.36 | 0.438 | Gcells/s | 0.984 | 4.3e-07 |
| bellman_recompute | 40000 | cpu | 1.37e+03 | 2.76e+03 | 2.0x | 1.17 | 0.579 | Gcells/s | 2.81 | 4.3e-07 |
| bellman_recompute | 60000 | cpu | 2.61e+03 | 5.41e+03 | 2.1x | 1.38 | 0.665 | Gcells/s | 5.24 | 4.3e-07 |
| bellman_recompute | 1000 | gpu | 0.084 | 0.159 | 1.9x | 11.9 | 6.28 | Gcells/s | 0.281 | 3.5e-07 |
| bellman_recompute | 2000 | gpu | 0.149 | 0.157 | 1.1x | 26.9 | 25.5 | Gcells/s | 0.379 | 3.5e-07 |
| bellman_recompute | 5000 | gpu | 0.24 | 0.401 | 1.7x | 104 | 62.4 | Gcells/s | 0.438 | 3.5e-07 |
| bellman_recompute | 10000 | gpu | 0.423 | 0.847 | 2.0x | 236 | 118 | Gcells/s | 0.641 | 3.5e-07 |
| bellman_recompute | 20000 | gpu | 1.09 | 2.56 | 2.4x | 367 | 156 | Gcells/s | 0.341 | 3.5e-07 |
| bellman_recompute | 40000 | gpu | 2.77 | 7.85 | 2.8x | 577 | 204 | Gcells/s | 0.385 | 3.5e-07 |
| bellman_recompute | 60000 | gpu | 4.7 | 13.9 | 3.0x | 766 | 259 | Gcells/s | 0.397 | 3.5e-07 |
| kalman | 1000 | cpu | 3.11 | 2.92 | 0.9x | 322 | 342 | ksteps/s | 0.23 | 1.1e-05 |
| kalman | 10000 | cpu | 35 | 33 | 0.9x | 286 | 303 | ksteps/s | 0.251 | 2.2e-06 |
| kalman | 100000 | cpu | 343 | 320 | 0.9x | 291 | 312 | ksteps/s | 0.547 | 5.7e-06 |
| kalman | 1000 | gpu | 259 | 282 | 1.1x | 3.87 | 3.55 | ksteps/s | 3.3 | 1.4e-05 |
| kalman | 10000 | gpu | 2.48e+03 | 2.69e+03 | 1.1x | 4.03 | 3.72 | ksteps/s | 4.74 | 2.0e-05 |
| kalman | 100000 | gpu | 2.46e+04 | 2.69e+04 | 1.1x | 4.06 | 3.71 | ksteps/s | 19.1 | 5.7e-06 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | 256 | not up to 8192 |
| matmul | float64 | gpu | 512 | not up to 8192 |
| bellman_stored | float32 | gpu | 1000 | 40000 |
| bellman_stored | float64 | gpu | 1000 | 20000 |
| bellman_recompute | float32 | gpu | 1000 | 40000 |
| bellman_recompute | float64 | gpu | 1000 | 20000 |
| kalman | float32 | gpu | not up to 100000 | not up to 100000 |
| kalman | float64 | gpu | not up to 100000 | not up to 100000 |

**Each GPU's results against this machine's CPU:** the largest relative difference between their samples

| workload | size | device | f32 | f64 |
|---|---|---|---|---|
| matmul | 256 | gpu | 0.0e+00 | 0.0e+00 |
| matmul | 512 | gpu | 0.0e+00 | 1.0e-15 |
| matmul | 1024 | gpu | 1.3e-06 | 2.2e-15 |
| matmul | 2048 | gpu | 1.8e-06 | 2.1e-15 |
| matmul | 4096 | gpu | 2.5e-06 | 3.2e-15 |
| matmul | 8192 | gpu | 3.3e-06 | 5.8e-15 |
| bellman_stored | 1000 | gpu | 3.8e-07 | 4.4e-16 |
| bellman_stored | 2000 | gpu | 2.5e-07 | 7.9e-16 |
| bellman_stored | 5000 | gpu | 2.2e-07 | 2.2e-16 |
| bellman_stored | 10000 | gpu | 2.4e-07 | 5.9e-16 |
| bellman_stored | 20000 | gpu | 2.3e-07 | 4.0e-16 |
| bellman_stored | 40000 | gpu | 2.1e-07 | 2.9e-16 |
| bellman_stored | 60000 | gpu | 2.3e-07 | 2.9e-16 |
| bellman_recompute | 1000 | gpu | 3.8e-07 | 4.4e-16 |
| bellman_recompute | 2000 | gpu | 2.5e-07 | 7.9e-16 |
| bellman_recompute | 5000 | gpu | 2.2e-07 | 2.2e-16 |
| bellman_recompute | 10000 | gpu | 2.4e-07 | 5.9e-16 |
| bellman_recompute | 20000 | gpu | 2.3e-07 | 4.0e-16 |
| bellman_recompute | 40000 | gpu | 2.1e-07 | 2.9e-16 |
| bellman_recompute | 60000 | gpu | 2.3e-07 | 2.9e-16 |
| kalman | 1000 | gpu | 2.5e-05 | 7.3e-08 |
| kalman | 10000 | gpu | 2.1e-05 | 6.0e-08 |
| kalman | 100000 | gpu | 9.3e-07 | 3.4e-08 |

## digitalocean-gpu-mi350x1-288gb-spot (highest)

AMD EPYC 9575F 64-Core Processor, 24 threads, 252 GiB; GPU: AMD Instinct MI350X VF, 294592 MiB; JAX 0.12.0

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.081 | 0.145 | 1.8x | 414 | 232 | GFLOPS | 1.07 | 5.3e-07 |
| matmul | 512 | gpu | 0.0806 | 0.151 | 1.9x | 3.33e+03 | 1.77e+03 | GFLOPS | 1.08 | 7.8e-07 |
| matmul | 1024 | gpu | 0.163 | 0.19 | 1.2x | 1.31e+04 | 1.13e+04 | GFLOPS | 1.09 | 9.7e-07 |
| matmul | 2048 | gpu | 0.258 | 0.381 | 1.5x | 6.65e+04 | 4.5e+04 | GFLOPS | 1.16 | 1.7e-06 |
| matmul | 4096 | gpu | 1.36 | 2.41 | 1.8x | 1.01e+05 | 5.71e+04 | GFLOPS | 39.8 | 2.7e-06 |
| matmul | 8192 | gpu | 8.2 | 15.7 | 1.9x | 1.34e+05 | 6.98e+04 | GFLOPS | 5.87 | 3.4e-06 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

