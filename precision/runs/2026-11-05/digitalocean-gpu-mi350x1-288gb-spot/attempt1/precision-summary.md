## digitalocean-gpu-mi350x1-288gb-spot

AMD EPYC 9575F 64-Core Processor, 24 threads, 252 GiB; GPU: AMD Instinct MI350X VF, 294592 MiB; JAX 0.12.0

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | cpu | 0.0925 | 0.102 | 1.1x | 363 | 329 | GFLOPS | 0.0499 | 5.3e-07 |
| matmul | 1024 | cpu | 0.632 | 1.15 | 1.8x | 3.4e+03 | 1.87e+03 | GFLOPS | 0.0521 | 5.6e-07 |
| matmul | 256 | gpu | 0.0779 | error |  | 431 |  | GFLOPS |  |  |
| matmul | 1024 | gpu | 0.148 | error |  | 1.45e+04 |  | GFLOPS |  |  |
| bellman_stored | 1000 | cpu | 0.0796 | 0.131 | 1.6x | 50.2 | 60.9 | GB/s | 0.059 | 4.3e-07 |
| bellman_stored | 5000 | cpu | 0.363 | 0.785 | 2.2x | 275 | 255 | GB/s | 0.0545 | 4.3e-07 |
| bellman_stored | 1000 | gpu | 0.0764 | 0.0823 | 1.1x | 52.3 | 97.2 | GB/s | 0.28 | 3.5e-07 |
| bellman_stored | 5000 | gpu | 0.0825 | 0.149 | 1.8x | 1.21e+03 | 1.35e+03 | GB/s | 0.317 | 3.5e-07 |
| bellman_recompute | 1000 | cpu | 0.373 | 1.78 | 4.8x | 2.68 | 0.56 | Gcells/s | 0.105 | 4.3e-07 |
| bellman_recompute | 5000 | cpu | 7.07 | 42.8 | 6.1x | 3.53 | 0.584 | Gcells/s | 0.129 | 4.3e-07 |
| bellman_recompute | 1000 | gpu | 0.0757 | 0.0839 | 1.1x | 13.2 | 11.9 | Gcells/s | 0.309 | 3.5e-07 |
| bellman_recompute | 5000 | gpu | 0.243 | 0.393 | 1.6x | 103 | 63.6 | Gcells/s | 0.462 | 3.5e-07 |
| kalman | 1000 | cpu | 3.11 | 2.92 | 0.9x | 321 | 343 | ksteps/s | 0.216 | 1.1e-05 |
| kalman | 1000 | gpu | error | error |  |  |  | ksteps/s |  |  |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | 256 | not up to 1024 |
| matmul | float64 | gpu | no comparison | no comparison |
| bellman_stored | float32 | gpu | 1000 | not up to 5000 |
| bellman_stored | float64 | gpu | 1000 | not up to 5000 |
| bellman_recompute | float32 | gpu | 1000 | not up to 5000 |
| bellman_recompute | float64 | gpu | 1000 | not up to 5000 |
| kalman | float32 | gpu | no comparison | no comparison |
| kalman | float64 | gpu | no comparison | no comparison |

**Each GPU's results against this machine's CPU:** the largest relative difference between their samples

| workload | size | device | f32 | f64 |
|---|---|---|---|---|
| matmul | 256 | gpu | 0.0e+00 |  |
| matmul | 1024 | gpu | 1.4e-06 |  |
| bellman_stored | 1000 | gpu | 3.8e-07 | 4.4e-16 |
| bellman_stored | 5000 | gpu | 2.2e-07 | 2.2e-16 |
| bellman_recompute | 1000 | gpu | 3.8e-07 | 4.4e-16 |
| bellman_recompute | 5000 | gpu | 2.2e-07 | 2.2e-16 |
| kalman | 1000 | gpu |  |  |

## digitalocean-gpu-mi350x1-288gb-spot (highest)

AMD EPYC 9575F 64-Core Processor, 24 threads, 252 GiB; GPU: AMD Instinct MI350X VF, 294592 MiB; JAX 0.12.0

| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate | unit | f64 cold s | f32 vs f64 |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | 256 | gpu | 0.0842 | error |  | 398 |  | GFLOPS |  |  |
| matmul | 1024 | gpu | 0.152 | error |  | 1.41e+04 |  | GFLOPS |  |  |

**Where the GPU first beats this machine's CPU** (warm, then cold)

| workload | precision | device | warm | cold |
|---|---|---|---|---|
| matmul | float32 | gpu | no comparison | no comparison |
| matmul | float64 | gpu | no comparison | no comparison |

