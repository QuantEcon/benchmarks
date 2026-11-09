### P1 `vfi`, grid sweep, float64, on aws-c7i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 1.2 (1.21) | 0.627 (2.36) | 0.135 (3.19) | 0.0663 (0.131) | jax-cpu | jax-cpu |
| 200 | 3.64 (3.57) | 3.01 (4.75) | 0.471 (3.53) | 0.157 (0.229) | jax-cpu | jax-cpu |
| 400 | 27.9 (27.9) | 14 (15.7) | 7.76 (10.2) | 0.505 (0.572) | jax-cpu | jax-cpu |
| 700 | 93.1 (93.1) | 46.1 (47.8) | 20.5 (23.2) | 1.82 (1.88) | jax-cpu | jax-cpu |
| 1,000 | 208 (206) | 95.4 (97.1) | 26.5 (24.2) | 3.65 (3.68) | jax-cpu | jax-cpu |
| 1,500 | timeout | 234 (235) | 50.9 (45.8) | 8.79 (8.82) | jax-cpu | jax-cpu |

### P3 `egm`, grid sweep, float64, on aws-c7i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 200 | 0.338 (0.344) | 0.489 (4.93) | 0.502 (5.11) | 0.648 (0.82) | numpy | numpy |
| 1,000 | 1.29 (1.29) | 2.27 (6.73) | 2.1 (6.69) | 3.04 (3.19) | numpy | numpy |
| 5,000 | 7.95 (7.98) | 12 (16.4) | 11.7 (16.1) | 17.8 (17.9) | numpy | numpy |
| 20,000 | 36.7 (36.6) | 51.1 (55.7) | 52.4 (56.3) | 81.5 (81.7) | numpy | numpy |
| 50,000 | 85.3 (84.9) | 137 (141) | 132 (138) | 216 (217) | numpy | numpy |

### P3 `egm`, grid sweep, float32, on aws-c7i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 200 | 0.291 (0.292) | 0.288 (4.81) | 0.347 (5.17) | 0.429 (0.614) | numba | numpy |
| 1,000 | 0.885 (0.887) | 1.58 (6.09) | 1.31 (6.07) | 2.56 (2.75) | numpy | numpy |
| 5,000 | 4.66 (4.66) | 7.47 (12) | 6.68 (11.4) | 12.8 (13.1) | numpy | numpy |
| 20,000 | 25.8 (26) | 32.1 (36.6) | 33.1 (37.3) | 64 (64.1) | numpy | numpy |
| 50,000 | 64.2 (64.4) | 91.9 (96.8) | 86 (89.7) | 172 (172) | numpy | numpy |

- 200, `numpy`: 2,208 iterations, against NumPy's 2,192 in float64
- 200, `numba`: 2,205 iterations, against NumPy's 2,192 in float64
- 200, `numba-parallel`: 2,205 iterations, against NumPy's 2,192 in float64
- 200, `jax-cpu`: 2,205 iterations, against NumPy's 2,192 in float64
- 1,000, `numpy`: 2,210 iterations, against NumPy's 2,190 in float64
- 1,000, `numba`: 2,206 iterations, against NumPy's 2,190 in float64
- 1,000, `numba-parallel`: 2,206 iterations, against NumPy's 2,190 in float64
- 1,000, `jax-cpu`: 2,206 iterations, against NumPy's 2,190 in float64
- 5,000, `numpy`: 2,211 iterations, against NumPy's 2,190 in float64
- 5,000, `numba`: 2,210 iterations, against NumPy's 2,190 in float64
- 5,000, `numba-parallel`: 2,210 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-cpu`: 2,210 iterations, against NumPy's 2,190 in float64
- 20,000, `numpy`: 2,210 iterations, against NumPy's 2,190 in float64
- 20,000, `numba`: 2,210 iterations, against NumPy's 2,190 in float64
- 20,000, `numba-parallel`: 2,210 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-cpu`: 2,210 iterations, against NumPy's 2,190 in float64
- 50,000, `numpy`: 2,213 iterations, against NumPy's 2,190 in float64
- 50,000, `numba`: 2,210 iterations, against NumPy's 2,190 in float64
- 50,000, `numba-parallel`: 2,210 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-cpu`: 2,210 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float64, on aws-c7i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00752 (0.00763) | 0.0167 (0.63) | 0.00199 (0.984) | 0.00892 (0.251) | numba-parallel | numpy |
| 10,000 | 0.0637 (0.0643) | 0.166 (0.784) | 0.0152 (0.987) | 0.0932 (0.334) | numba-parallel | numpy |
| 100,000 | 0.608 (0.609) | 1.67 (2.28) | 0.096 (1.08) | 1 (1.23) | numba-parallel | numpy |
| 1,000,000 | 6.23 (6.24) | 16.5 (17.1) | 1.31 (2.29) | 10.7 (11) | numba-parallel | numba-parallel |
| 10,000,000 | 87 (87) | 165 (166) | 9.59 (10.8) | 65.6 (65.9) | numba-parallel | numba-parallel |

### P4 `wealth`, households sweep, float32, on aws-c7i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00691 (0.00704) | 0.0143 (0.624) | 0.00183 (0.974) | 0.00273 (0.218) | numba-parallel | numpy |
| 10,000 | 0.0497 (0.0501) | 0.144 (0.754) | 0.0134 (0.986) | 0.0292 (0.241) | numba-parallel | numpy |
| 100,000 | 0.504 (0.523) | 1.44 (2.05) | 0.0873 (1.06) | 0.382 (0.587) | numba-parallel | numpy |
| 1,000,000 | 5.08 (5.14) | 14.4 (15) | 0.904 (1.88) | 4.48 (4.69) | numba-parallel | numba-parallel |
| 10,000,000 | 62.2 (62.3) | 144 (144) | 13.2 (14.2) | 52.5 (52.7) | numba-parallel | numba-parallel |

### P7 `kalman`, batch sweep, float64, on aws-c7i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0284 (0.0287) | 0.00357 (8.89) | 0.0037 (13.6) | 0.005 (0.312) | numba | numpy |
| 10 | 0.0485 (0.0489) | 0.0366 (8.92) | 0.0284 (13.7) | 0.0225 (0.33) | jax-cpu | numpy |
| 100 | 0.242 (0.243) | 0.357 (9.26) | 0.315 (13.9) | 0.207 (0.521) | jax-cpu | numpy |
| 1,000 | 2.06 (2.13) | 3.57 (12.5) | 3.21 (16.8) | 1.35 (1.68) | jax-cpu | jax-cpu |
| 10,000 | 19.2 (19.5) | 35.4 (44.3) | 28.9 (42.4) | 20.3 (20.6) | numpy | numpy |

### P7 `kalman`, batch sweep, float32, on aws-c7i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0319 (0.0323) | 0.0033 (8.91) | 0.00336 (13.6) | 0.00489 (0.308) | numba | numpy |
| 10 | 0.0512 (0.0516) | 0.0333 (8.91) | 0.029 (13.7) | 0.0209 (0.328) | jax-cpu | numpy |
| 100 | 0.237 (0.239) | 0.332 (9.2) | 0.296 (13.9) | 0.166 (0.478) | jax-cpu | numpy |
| 1,000 | 1.89 (1.9) | 3.29 (12.2) | 2.93 (16.5) | 1.22 (1.54) | jax-cpu | jax-cpu |
| 10,000 | 18.2 (18.2) | 33 (41.8) | 29.5 (43.1) | 24.6 (24.8) | numpy | numpy |

### P7 `kalman`, state sweep, float64, on aws-c7i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0271 (0.0273) | 0.00264 (8.88) | 0.00274 (13.6) | 0.00203 (0.296) | jax-cpu | numpy |
| 5 | 0.0279 (0.0283) | 0.00298 (8.92) | 0.00306 (13.5) | 0.00379 (0.303) | numba | numpy |
| 10 | 0.0284 (0.0287) | 0.0036 (8.89) | 0.0036 (13.7) | 0.00502 (0.312) | numba | numpy |
| 20 | 0.0313 (0.0317) | 0.00561 (8.89) | 0.00571 (13.6) | 0.00736 (0.311) | numba | numpy |
| 50 | 0.0604 (0.0607) | 0.022 (8.93) | 0.0218 (13.6) | 0.0302 (0.34) | numba-parallel | numpy |
| 100 | 0.142 (0.142) | 0.0913 (9.01) | 0.0905 (13.7) | 0.134 (0.448) | numba-parallel | numpy |
| 200 | 0.317 (0.318) | 0.268 (9.16) | 0.258 (13.9) | 0.713 (1.01) | numba-parallel | numpy |

### P7 `kalman`, state sweep, float32, on aws-c7i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0307 (0.031) | 0.00266 (8.89) | 0.00278 (13.6) | 0.002 (0.291) | jax-cpu | numpy |
| 5 | 0.0311 (0.0315) | 0.00294 (8.89) | 0.00303 (13.5) | 0.00507 (0.324) | numba | numpy |
| 10 | 0.0319 (0.0323) | 0.00337 (8.87) | 0.00347 (13.5) | 0.00487 (0.307) | numba | numpy |
| 20 | 0.0335 (0.0338) | 0.00508 (8.87) | 0.00497 (13.6) | 0.00664 (0.311) | numba-parallel | numpy |
| 50 | 0.045 (0.0453) | 0.0154 (8.88) | 0.0155 (13.5) | 0.022 (0.331) | numba | numpy |
| 100 | 0.112 (0.112) | 0.0586 (8.93) | 0.0585 (13.6) | 0.0919 (0.412) | numba-parallel | numpy |
| 200 | 0.213 (0.213) | 0.177 (9.07) | 0.183 (13.8) | 0.376 (0.67) | numba | numpy |

### P1 `vfi`, grid sweep, float64, on aws-c7i.8xlarge (timeout-1800)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,500 | 481 (482) | numpy | numpy |

### P1 `vfi`, grid sweep, float64, on aws-c7i.8xlarge (ynn-reduce-off)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 37.7 (37.8) | jax-cpu | jax-cpu |
| 1,500 | 86.2 (86.2) | jax-cpu | jax-cpu |

