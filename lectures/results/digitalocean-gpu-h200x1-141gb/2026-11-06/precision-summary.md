## digitalocean-gpu-h200x1-141gb

INTEL(R) XEON(R) PLATINUM 8592+, 24 threads, 236 GiB; GPU: NVIDIA H200, 143771 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | cpu | 0.175 | 0.152 | 0.9x | 192 | 220 | GFLOPS | 0.0993 | 5.3e-07 |
| matmul | 512 | cpu | 0.304 | 0.387 | 1.3x | 884 | 694 | GFLOPS | 0.102 | 7.8e-07 |
| matmul | 1024 | cpu | 1.06 | 1.57 | 1.5x | 2.04e+03 | 1.36e+03 | GFLOPS | 0.102 | 5.6e-07 |
| matmul | 2048 | cpu | 5.26 | 18.7 | 3.6x | 3.27e+03 | 920 | GFLOPS | 0.12 | 5.2e-07 |
| matmul | 4096 | cpu | 55.5 | 95.2 | 1.7x | 2.47e+03 | 1.44e+03 | GFLOPS | 0.187 | 3.6e-07 |
| matmul | 8192 | cpu | 341 | 646 | 1.9x | 3.22e+03 | 1.7e+03 | GFLOPS | 0.723 | 2.4e-07 |
| matmul | 256 | gpu | 0.087 | 0.106 | 1.2x | 386 | 316 | GFLOPS | 0.768 | 5.3e-05 |
| matmul | 512 | gpu | 0.0825 | 0.0885 | 1.1x | 3.25e+03 | 3.03e+03 | GFLOPS | 0.893 | 3.9e-05 |
| matmul | 1024 | gpu | 0.111 | 0.249 | 2.2x | 1.93e+04 | 8.61e+03 | GFLOPS | 0.897 | 3.5e-05 |
| matmul | 2048 | gpu | 0.217 | 0.453 | 2.1x | 7.91e+04 | 3.79e+04 | GFLOPS | 0.908 | 2.9e-05 |
| matmul | 4096 | gpu | 0.522 | 2.36 | 4.5x | 2.63e+05 | 5.83e+04 | GFLOPS | 0.967 | 3.8e-05 |
| matmul | 8192 | gpu | 2.76 | 17.6 | 6.4x | 3.98e+05 | 6.25e+04 | GFLOPS | 1.56 | 6.4e-05 |
| bellman_stored | 1000 | cpu | 0.197 | 0.213 | 1.1x | 20.3 | 37.5 | GB/s | 0.111 | 4.3e-07 |
| bellman_stored | 2000 | cpu | 0.171 | 0.287 | 1.7x | 93.6 | 112 | GB/s | 0.111 | 4.3e-07 |
| bellman_stored | 5000 | cpu | 0.489 | 0.893 | 1.8x | 205 | 224 | GB/s | 0.102 | 4.3e-07 |
| bellman_stored | 10000 | cpu | 1.96 | 4.19 | 2.1x | 204 | 191 | GB/s | 0.113 | 4.3e-07 |
| bellman_stored | 20000 | cpu | 8.46 | 556 | 65.7x | 189 | 5.75 | GB/s | 0.702 | 4.3e-07 |
| bellman_stored | 40000 | cpu | 1.16e+03 | 745 | 0.6x | 5.52 | 17.2 | GB/s | 0.875 | 4.3e-07 |
| bellman_stored | 60000 | cpu | 1.27e+03 | 1.09e+03 | 0.9x | 11.4 | 26.5 | GB/s | 1.19 | 4.3e-07 |
| bellman_stored | 1000 | gpu | 0.0877 | 0.166 | 1.9x | 45.6 | 48.1 | GB/s | 0.618 | 4.3e-07 |
| bellman_stored | 2000 | gpu | 0.166 | 0.197 | 1.2x | 96.4 | 162 | GB/s | 0.756 | 4.3e-07 |
| bellman_stored | 5000 | gpu | 0.246 | 0.233 | 0.9x | 407 | 858 | GB/s | 0.555 | 4.3e-07 |
| bellman_stored | 10000 | gpu | 0.276 | 0.39 | 1.4x | 1.45e+03 | 2.05e+03 | GB/s | 0.551 | 4.3e-07 |
| bellman_stored | 20000 | gpu | 0.548 | 0.915 | 1.7x | 2.92e+03 | 3.5e+03 | GB/s | 0.694 | 4.3e-07 |
| bellman_stored | 40000 | gpu | 1.64 | 3.26 | 2.0x | 3.91e+03 | 3.92e+03 | GB/s | 1.11 | 4.3e-07 |
| bellman_stored | 60000 | gpu | 3.5 | 6.92 | 2.0x | 4.11e+03 | 4.16e+03 | GB/s | 1.4 | 4.3e-07 |
| bellman_recompute | 1000 | cpu | 0.375 | 1.55 | 4.1x | 2.67 | 0.647 | Gcells/s | 0.142 | 4.3e-07 |
| bellman_recompute | 2000 | cpu | 1.14 | 5.89 | 5.2x | 3.51 | 0.679 | Gcells/s | 0.167 | 4.3e-07 |
| bellman_recompute | 5000 | cpu | 6.95 | 31.3 | 4.5x | 3.6 | 0.797 | Gcells/s | 0.2 | 4.3e-07 |
| bellman_recompute | 10000 | cpu | 24.2 | 149 | 6.1x | 4.13 | 0.672 | Gcells/s | 0.318 | 4.3e-07 |
| bellman_recompute | 20000 | cpu | 85.1 | 1.48e+03 | 17.4x | 4.7 | 0.269 | Gcells/s | 1.69 | 4.3e-07 |
| bellman_recompute | 40000 | cpu | 2.41e+03 | 4.16e+03 | 1.7x | 0.665 | 0.385 | Gcells/s | 4.33 | 4.3e-07 |
| bellman_recompute | 60000 | cpu | 3.99e+03 | 7.7e+03 | 1.9x | 0.903 | 0.468 | Gcells/s | 7.79 | 4.3e-07 |
| bellman_recompute | 1000 | gpu | 0.0928 | 0.0976 | 1.1x | 10.8 | 10.3 | Gcells/s | 0.699 | 4.3e-07 |
| bellman_recompute | 2000 | gpu | 0.146 | 0.175 | 1.2x | 27.5 | 22.8 | Gcells/s | 0.884 | 4.3e-07 |
| bellman_recompute | 5000 | gpu | 0.216 | 0.346 | 1.6x | 116 | 72.3 | Gcells/s | 0.856 | 4.3e-07 |
| bellman_recompute | 10000 | gpu | 0.432 | 0.726 | 1.7x | 232 | 138 | Gcells/s | 1.18 | 4.3e-07 |
| bellman_recompute | 20000 | gpu | 1.12 | 2.3 | 2.1x | 358 | 174 | Gcells/s | 1.99 | 4.3e-07 |
| bellman_recompute | 40000 | gpu | 3.3 | 8.35 | 2.5x | 485 | 192 | Gcells/s | 84.4 | 4.3e-07 |
| bellman_recompute | 60000 | gpu | 5.81 | 16.8 | 2.9x | 620 | 214 | Gcells/s | 82.6 | 4.3e-07 |
| kalman | 1000 | cpu | 4.43 | 3.96 | 0.9x | 226 | 252 | ksteps/s | 0.335 | 1.9e-05 |
| kalman | 10000 | cpu | 50 | 45.1 | 0.9x | 200 | 222 | ksteps/s | 0.382 | 1.7e-06 |
| kalman | 100000 | cpu | 496 | 456 | 0.9x | 202 | 219 | ksteps/s | 0.829 | 5.7e-06 |
| kalman | 1000 | gpu | 85.1 | 86.1 | 1.0x | 11.7 | 11.6 | ksteps/s | 0.793 | 5.4e-02 |
| kalman | 10000 | gpu | 837 | 881 | 1.1x | 12 | 11.4 | ksteps/s | 1.59 | 1.9e-02 |
| kalman | 100000 | gpu | 8.51e+03 | 8.7e+03 | 1.0x | 11.8 | 11.5 | ksteps/s | 9.55 | 2.2e-03 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | 256 | not up to 8192 |
| matmul | float64 | gpu | 256 | not up to 8192 |
| bellman_stored | float32 | gpu | 1000 | 40000 |
| bellman_stored | float64 | gpu | 1000 | 20000 |
| bellman_recompute | float32 | gpu | 1000 | 60000 |
| bellman_recompute | float64 | gpu | 1000 | not up to 60000 |
| kalman | float32 | gpu | not up to 100000 | not up to 100000 |
| kalman | float64 | gpu | not up to 100000 | not up to 100000 |

**Each GPU's results against this machine's CPU:** the largest relative difference between their samples

| workload | size | device | f32 | f64 |
|---|---|---|---|---|
| matmul | 256 | gpu | 5.3e-05 | 0.0e+00 |
| matmul | 512 | gpu | 3.8e-05 | 0.0e+00 |
| matmul | 1024 | gpu | 3.5e-05 | 1.3e-15 |
| matmul | 2048 | gpu | 2.9e-05 | 2.5e-15 |
| matmul | 4096 | gpu | 3.8e-05 | 3.7e-15 |
| matmul | 8192 | gpu | 6.5e-05 | 5.6e-15 |
| bellman_stored | 1000 | gpu | 2.9e-07 | 4.4e-16 |
| bellman_stored | 2000 | gpu | 2.8e-07 | 7.9e-16 |
| bellman_stored | 5000 | gpu | 1.6e-07 | 2.2e-16 |
| bellman_stored | 10000 | gpu | 2.4e-07 | 5.9e-16 |
| bellman_stored | 20000 | gpu | 1.6e-07 | 4.0e-16 |
| bellman_stored | 40000 | gpu | 2.8e-07 | 2.9e-16 |
| bellman_stored | 60000 | gpu | 1.9e-07 | 2.9e-16 |
| bellman_recompute | 1000 | gpu | 2.9e-07 | 4.4e-16 |
| bellman_recompute | 2000 | gpu | 2.8e-07 | 7.9e-16 |
| bellman_recompute | 5000 | gpu | 1.6e-07 | 2.2e-16 |
| bellman_recompute | 10000 | gpu | 2.4e-07 | 5.9e-16 |
| bellman_recompute | 20000 | gpu | 1.6e-07 | 4.0e-16 |
| bellman_recompute | 40000 | gpu | 2.8e-07 | 2.9e-16 |
| bellman_recompute | 60000 | gpu | 1.9e-07 | 2.9e-16 |
| kalman | 1000 | gpu | 5.5e-02 | 8.7e-07 |
| kalman | 10000 | gpu | 1.9e-02 | 1.4e-07 |
| kalman | 100000 | gpu | 2.2e-03 | 8.4e-08 |

## digitalocean-gpu-h200x1-141gb (highest)

INTEL(R) XEON(R) PLATINUM 8592+, 24 threads, 236 GiB; GPU: NVIDIA H200, 143771 MiB; JAX 0.12.0.1

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.0845 | 0.0897 | 1.1x | 397 | 374 | GFLOPS | 0.756 | 2.0e-07 |
| matmul | 512 | gpu | 0.105 | 0.0911 | 0.9x | 2.55e+03 | 2.95e+03 | GFLOPS | 0.906 | 2.3e-07 |
| matmul | 1024 | gpu | 0.253 | 0.199 | 0.8x | 8.5e+03 | 1.08e+04 | GFLOPS | 0.9 | 2.3e-07 |
| matmul | 2048 | gpu | 0.54 | 0.484 | 0.9x | 3.18e+04 | 3.55e+04 | GFLOPS | 0.907 | 1.7e-06 |
| matmul | 4096 | gpu | 2.86 | 2.34 | 0.8x | 4.81e+04 | 5.87e+04 | GFLOPS | 0.991 | 2.7e-06 |
| matmul | 8192 | gpu | 22 | 17.5 | 0.8x | 5.01e+04 | 6.28e+04 | GFLOPS | 1.54 | 1.1e-06 |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

