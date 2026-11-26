## digitalocean-gpu-mi300x1-192gb

INTEL(R) XEON(R) PLATINUM 8568Y+, 20 threads, 236 GiB; GPU: AMD Instinct MI300X VF, 196288 MiB; JAX 0.12.0

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | cpu | 0.176 | 0.163 | 0.9x | 190 | 205 | GFLOPS | 0.0799 | 5.3e-07 |
| matmul | 512 | cpu | 0.263 | 0.335 | 1.3x | 1.02e+03 | 801 | GFLOPS | 0.0736 | 7.8e-07 |
| matmul | 1024 | cpu | 0.918 | 1.77 | 1.9x | 2.34e+03 | 1.21e+03 | GFLOPS | 0.0769 | 5.6e-07 |
| matmul | 2048 | cpu | 5.49 | 21.2 | 3.9x | 3.13e+03 | 809 | GFLOPS | 0.0991 | 5.2e-07 |
| matmul | 4096 | cpu | 57.2 | 99.4 | 1.7x | 2.4e+03 | 1.38e+03 | GFLOPS | 0.173 | 3.6e-07 |
| matmul | 8192 | cpu | 358 | 705 | 2.0x | 3.07e+03 | 1.56e+03 | GFLOPS | 0.848 | 2.4e-07 |
| matmul | 256 | gpu | 0.0868 | 0.15 | 1.7x | 387 | 223 | GFLOPS | 6.93 | 7.1e-04 |
| matmul | 512 | gpu | 0.114 | 0.161 | 1.4x | 2.36e+03 | 1.67e+03 | GFLOPS | 6.88 | 7.0e-04 |
| matmul | 1024 | gpu | 0.158 | 0.254 | 1.6x | 1.36e+04 | 8.45e+03 | GFLOPS | 6.87 | 6.8e-04 |
| matmul | 2048 | gpu | 0.262 | 1.13 | 4.3x | 6.56e+04 | 1.52e+04 | GFLOPS | 6.76 | 6.7e-04 |
| matmul | 4096 | gpu | 0.919 | 6.48 | 7.0x | 1.49e+05 | 2.12e+04 | GFLOPS | 6.72 | 6.7e-04 |
| matmul | 8192 | gpu | 4.83 | 54 | 11.2x | 2.28e+05 | 2.03e+04 | GFLOPS | 6.79 | 6.6e-04 |
| bellman_stored | 1000 | cpu | 0.164 | 0.124 | 0.8x | 24.4 | 64.7 | GB/s | 0.086 | 4.3e-07 |
| bellman_stored | 2000 | cpu | 0.197 | 0.2 | 1.0x | 81.2 | 160 | GB/s | 0.0864 | 4.3e-07 |
| bellman_stored | 5000 | cpu | 0.461 | 0.87 | 1.9x | 217 | 230 | GB/s | 0.0774 | 4.3e-07 |
| bellman_stored | 10000 | cpu | 2.07 | 4.64 | 2.2x | 193 | 172 | GB/s | 0.0833 | 4.3e-07 |
| bellman_stored | 20000 | cpu | 9.23 | 520 | 56.3x | 173 | 6.16 | GB/s | 0.623 | 4.3e-07 |
| bellman_stored | 40000 | cpu | 1.06e+03 | 717 | 0.7x | 6.03 | 17.9 | GB/s | 0.758 | 4.3e-07 |
| bellman_stored | 60000 | cpu | 1.2e+03 | 1.03e+03 | 0.9x | 12 | 28 | GB/s | 1.11 | 4.3e-07 |
| bellman_stored | 1000 | gpu | 0.145 | 0.0837 | 0.6x | 27.6 | 95.6 | GB/s | 0.317 | 3.5e-07 |
| bellman_stored | 2000 | gpu | 0.147 | 0.0828 | 0.6x | 109 | 386 | GB/s | 0.304 | 3.5e-07 |
| bellman_stored | 5000 | gpu | 0.155 | 0.155 | 1.0x | 643 | 1.29e+03 | GB/s | 0.332 | 3.5e-07 |
| bellman_stored | 10000 | gpu | 0.249 | 0.322 | 1.3x | 1.61e+03 | 2.49e+03 | GB/s | 0.346 | 3.5e-07 |
| bellman_stored | 20000 | gpu | 0.681 | 1.17 | 1.7x | 2.35e+03 | 2.73e+03 | GB/s | 0.365 | 3.5e-07 |
| bellman_stored | 40000 | gpu | 2.23 | 4.25 | 1.9x | 2.87e+03 | 3.01e+03 | GB/s | 0.593 | 3.5e-07 |
| bellman_stored | 60000 | gpu | 4.58 | 9.18 | 2.0x | 3.14e+03 | 3.14e+03 | GB/s | 1.09 | 3.5e-07 |
| bellman_recompute | 1000 | cpu | 0.445 | 1.75 | 3.9x | 2.25 | 0.573 | Gcells/s | 0.128 | 4.3e-07 |
| bellman_recompute | 2000 | cpu | 1.29 | 5.96 | 4.6x | 3.1 | 0.671 | Gcells/s | 0.153 | 4.3e-07 |
| bellman_recompute | 5000 | cpu | 7.04 | 34.6 | 4.9x | 3.55 | 0.723 | Gcells/s | 0.163 | 4.3e-07 |
| bellman_recompute | 10000 | cpu | 20.4 | 116 | 5.7x | 4.91 | 0.865 | Gcells/s | 0.263 | 4.3e-07 |
| bellman_recompute | 20000 | cpu | 70.9 | 1.39e+03 | 19.6x | 5.64 | 0.288 | Gcells/s | 1.56 | 4.3e-07 |
| bellman_recompute | 40000 | cpu | 2.24e+03 | 4.05e+03 | 1.8x | 0.713 | 0.395 | Gcells/s | 4.16 | 4.3e-07 |
| bellman_recompute | 60000 | cpu | 3.63e+03 | 7.62e+03 | 2.1x | 0.992 | 0.472 | Gcells/s | 7.67 | 4.3e-07 |
| bellman_recompute | 1000 | gpu | 0.147 | 0.156 | 1.1x | 6.78 | 6.42 | Gcells/s | 0.336 | 3.5e-07 |
| bellman_recompute | 2000 | gpu | 0.163 | 0.166 | 1.0x | 24.5 | 24.1 | Gcells/s | 0.411 | 3.5e-07 |
| bellman_recompute | 5000 | gpu | 0.247 | 0.473 | 1.9x | 101 | 52.9 | Gcells/s | 0.493 | 3.5e-07 |
| bellman_recompute | 10000 | gpu | 0.562 | 0.92 | 1.6x | 178 | 109 | Gcells/s | 0.596 | 3.5e-07 |
| bellman_recompute | 20000 | gpu | 1.17 | 2.81 | 2.4x | 342 | 142 | Gcells/s | 0.475 | 3.5e-07 |
| bellman_recompute | 40000 | gpu | 3.39 | 8.46 | 2.5x | 471 | 189 | Gcells/s | 0.501 | 3.5e-07 |
| bellman_recompute | 60000 | gpu | 5.98 | 16.4 | 2.7x | 602 | 220 | Gcells/s | 0.683 | 3.5e-07 |
| kalman | 1000 | cpu | 4.13 | 4.37 | 1.1x | 242 | 229 | ksteps/s | 0.317 | 1.9e-05 |
| kalman | 10000 | cpu | 47.3 | 45.1 | 1.0x | 211 | 222 | ksteps/s | 0.348 | 1.7e-06 |
| kalman | 100000 | cpu | 492 | 460 | 0.9x | 203 | 217 | ksteps/s | 0.747 | 5.7e-06 |
| kalman | 1000 | gpu | 140 | 160 | 1.1x | 7.15 | 6.26 | ksteps/s | 19.5 | 1.4e-05 |
| kalman | 10000 | gpu | 1.33e+03 | 1.56e+03 | 1.2x | 7.53 | 6.39 | ksteps/s | 20.7 | 4.9e-03 |
| kalman | 100000 | gpu | 1.3e+04 | 1.54e+04 | 1.2x | 7.67 | 6.5 | ksteps/s | 35.3 | 1.9e-03 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | 256 | not up to 8192 |
| matmul | float64 | gpu | 256 | not up to 8192 |
| bellman_stored | float32 | gpu | 1000 | 40000 |
| bellman_stored | float64 | gpu | 1000 | 20000 |
| bellman_recompute | float32 | gpu | 1000 | 40000 |
| bellman_recompute | float64 | gpu | 1000 | 20000 |
| kalman | float32 | gpu | not up to 100000 | not up to 100000 |
| kalman | float64 | gpu | not up to 100000 | not up to 100000 |

**Each GPU's results against this machine's CPU:** the largest relative difference between their samples

| workload | size | device | f32 | f64 |
|---|---|---|---|---|
| matmul | 256 | gpu | 7.1e-04 | 0.0e+00 |
| matmul | 512 | gpu | 7.0e-04 | 4.3e-16 |
| matmul | 1024 | gpu | 6.8e-04 | 1.6e-15 |
| matmul | 2048 | gpu | 6.7e-04 | 1.5e-15 |
| matmul | 4096 | gpu | 6.7e-04 | 4.5e-15 |
| matmul | 8192 | gpu | 6.6e-04 | 4.8e-15 |
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
| kalman | 1000 | gpu | 5.0e-06 | 7.3e-08 |
| kalman | 10000 | gpu | 4.9e-03 | 6.0e-08 |
| kalman | 100000 | gpu | 1.9e-03 | 3.4e-08 |

## digitalocean-gpu-mi300x1-192gb (autotune0)

INTEL(R) XEON(R) PLATINUM 8568Y+, 20 threads, 236 GiB; GPU: AMD Instinct MI300X VF, 196288 MiB; JAX 0.12.0

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.0951 | 0.157 | 1.6x | 353 | 214 | GFLOPS | 6.88 | 7.1e-04 |
| matmul | 512 | gpu | 0.161 | 0.159 | 1.0x | 1.67e+03 | 1.69e+03 | GFLOPS | 6.99 | 7.0e-04 |
| matmul | 1024 | gpu | 0.168 | 0.251 | 1.5x | 1.28e+04 | 8.55e+03 | GFLOPS | 6.83 | 6.8e-04 |
| matmul | 2048 | gpu | 0.612 | 1.18 | 1.9x | 2.81e+04 | 1.45e+04 | GFLOPS | 6.79 | 6.7e-04 |
| matmul | 4096 | gpu | 3.66 | 6.5 | 1.8x | 3.76e+04 | 2.11e+04 | GFLOPS | 6.91 | 6.7e-04 |
| matmul | 8192 | gpu | 27.7 | 54.2 | 2.0x | 3.96e+04 | 2.03e+04 | GFLOPS | 6.84 | 6.6e-04 |
| matmul | 16384 | gpu | 307 | 500 | 1.6x | 2.86e+04 | 1.76e+04 | GFLOPS | 7.46 | 6.6e-04 |
| bellman_stored | 1000 | gpu | 0.0917 | 0.125 | 1.4x | 43.6 | 63.9 | GB/s | 0.237 | 3.5e-07 |
| bellman_stored | 2000 | gpu | 0.16 | 0.096 | 0.6x | 100 | 333 | GB/s | 0.239 | 3.5e-07 |
| bellman_stored | 5000 | gpu | 0.162 | 0.157 | 1.0x | 619 | 1.27e+03 | GB/s | 0.242 | 3.5e-07 |
| bellman_stored | 10000 | gpu | 0.254 | 0.351 | 1.4x | 1.58e+03 | 2.28e+03 | GB/s | 0.248 | 3.5e-07 |
| bellman_stored | 20000 | gpu | 0.878 | 1.19 | 1.4x | 1.82e+03 | 2.69e+03 | GB/s | 0.238 | 3.5e-07 |
| bellman_stored | 40000 | gpu | 2.51 | 4.47 | 1.8x | 2.55e+03 | 2.86e+03 | GB/s | 0.236 | 3.5e-07 |
| bellman_stored | 60000 | gpu | 4.99 | 9.3 | 1.9x | 2.88e+03 | 3.1e+03 | GB/s | 0.241 | 3.5e-07 |
| bellman_recompute | 1000 | gpu | 0.182 | 0.17 | 0.9x | 5.5 | 5.89 | Gcells/s | 0.277 | 3.5e-07 |
| bellman_recompute | 2000 | gpu | 0.226 | 0.308 | 1.4x | 17.7 | 13 | Gcells/s | 0.315 | 3.5e-07 |
| bellman_recompute | 5000 | gpu | 0.248 | 0.497 | 2.0x | 101 | 50.3 | Gcells/s | 0.308 | 3.5e-07 |
| bellman_recompute | 10000 | gpu | 0.666 | 1.07 | 1.6x | 150 | 93.4 | Gcells/s | 0.394 | 3.5e-07 |
| bellman_recompute | 20000 | gpu | 1.16 | 3 | 2.6x | 345 | 133 | Gcells/s | 0.298 | 3.5e-07 |
| bellman_recompute | 40000 | gpu | 3.84 | 9.23 | 2.4x | 417 | 173 | Gcells/s | 0.336 | 3.5e-07 |
| bellman_recompute | 60000 | gpu | 6.63 | 17.4 | 2.6x | 543 | 207 | Gcells/s | 0.336 | 3.5e-07 |
| kalman | 1000 | gpu | 241 | 265 | 1.1x | 4.15 | 3.77 | ksteps/s | 19.8 | 1.4e-05 |
| kalman | 10000 | gpu | 2.32e+03 | 2.57e+03 | 1.1x | 4.31 | 3.89 | ksteps/s | 22.3 | 2.0e-05 |
| kalman | 100000 | gpu | 2.31e+04 | 2.55e+04 | 1.1x | 4.33 | 3.92 | ksteps/s | 45.2 | 5.7e-06 |

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

## digitalocean-gpu-mi300x1-192gb (highest)

INTEL(R) XEON(R) PLATINUM 8568Y+, 20 threads, 236 GiB; GPU: AMD Instinct MI300X VF, 196288 MiB; JAX 0.12.0

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.139 | 0.153 | 1.1x | 242 | 219 | GFLOPS | 6.82 | 5.3e-07 |
| matmul | 512 | gpu | 0.146 | 0.166 | 1.1x | 1.84e+03 | 1.62e+03 | GFLOPS | 6.97 | 1.0e-06 |
| matmul | 1024 | gpu | 0.155 | 0.234 | 1.5x | 1.39e+04 | 9.16e+03 | GFLOPS | 6.89 | 1.1e-06 |
| matmul | 2048 | gpu | 0.331 | 1.2 | 3.6x | 5.2e+04 | 1.44e+04 | GFLOPS | 6.83 | 1.2e-06 |
| matmul | 4096 | gpu | 1.4 | 6.52 | 4.6x | 9.8e+04 | 2.11e+04 | GFLOPS | 6.85 | 2.5e-06 |
| matmul | 8192 | gpu | 9.25 | 54.3 | 5.9x | 1.19e+05 | 2.03e+04 | GFLOPS | 6.91 | 3.1e-06 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

