### P1 `vfi`, grid sweep, float64, on aws-c8i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 0.858 (0.864) | 0.519 (1.88) | 0.111 (2.52) | 0.0562 (0.11) | jax-cpu | jax-cpu |
| 200 | 3.07 (3.1) | 2.48 (3.86) | 0.394 (2.8) | 0.131 (0.191) | jax-cpu | jax-cpu |
| 400 | 20.7 (20.5) | 12.3 (13.8) | 6.58 (8.48) | 0.397 (0.453) | jax-cpu | jax-cpu |
| 700 | 69 (68.8) | 37.7 (39.1) | 14.7 (16.4) | 1.34 (1.41) | jax-cpu | jax-cpu |
| 1,000 | 156 (156) | 77.5 (78.9) | 21.1 (23.4) | 2.84 (2.87) | jax-cpu | jax-cpu |
| 1,500 | 362 (362) | 204 (208) | 50.1 (44.4) | 6.91 (6.91) | jax-cpu | jax-cpu |

### P3 `egm`, grid sweep, float64, on aws-c8i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 200 | 0.27 (0.271) | 0.357 (3.87) | 0.411 (3.99) | 0.509 (0.639) | numpy | numpy |
| 1,000 | 0.91 (0.908) | 1.78 (5.29) | 1.62 (5.21) | 2.42 (2.54) | numpy | numpy |
| 5,000 | 6.3 (6.29) | 9.79 (13.3) | 9.49 (12.6) | 15.8 (16) | numpy | numpy |
| 20,000 | 32.2 (31.9) | 43.1 (50.6) | 41.6 (45.8) | 76.6 (76.7) | numpy | numpy |
| 50,000 | 76.1 (76.1) | 119 (122) | 115 (119) | 203 (204) | numpy | numpy |

### P3 `egm`, grid sweep, float32, on aws-c8i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 200 | 0.239 (0.239) | 0.241 (3.81) | 0.294 (4.06) | 0.353 (0.504) | numpy | numpy |
| 1,000 | 0.7 (0.7) | 1.18 (4.78) | 1.05 (4.84) | 1.7 (1.84) | numpy | numpy |
| 5,000 | 3.81 (3.78) | 6.35 (9.96) | 5.58 (9.47) | 11.1 (11.3) | numpy | numpy |
| 20,000 | 23.4 (23.4) | 27.3 (31) | 27.6 (28.7) | 59 (59.1) | numpy | numpy |
| 50,000 | 58.5 (59) | 81.7 (85.8) | 69 (72.4) | 158 (159) | numpy | numpy |

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

### P4 `wealth`, households sweep, float64, on aws-c8i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00614 (0.00641) | 0.013 (0.501) | 0.0016 (0.782) | 0.00713 (0.2) | numba-parallel | numpy |
| 10,000 | 0.0496 (0.0495) | 0.132 (0.629) | 0.0118 (0.803) | 0.0729 (0.264) | numba-parallel | numpy |
| 100,000 | 0.487 (0.487) | 1.31 (1.81) | 0.0786 (0.868) | 0.83 (1.03) | numba-parallel | numpy |
| 1,000,000 | 5.13 (5.13) | 13.1 (13.5) | 0.78 (1.58) | 9.1 (9.29) | numba-parallel | numba-parallel |
| 10,000,000 | 69.5 (69.6) | 132 (132) | 7.81 (8.58) | 52.5 (53.2) | numba-parallel | numba-parallel |

### P4 `wealth`, households sweep, float32, on aws-c8i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.0056 (0.00598) | 0.0114 (0.507) | 0.00145 (0.79) | 0.00214 (0.173) | numba-parallel | numpy |
| 10,000 | 0.0408 (0.0407) | 0.115 (0.61) | 0.0108 (0.787) | 0.0233 (0.194) | numba-parallel | numpy |
| 100,000 | 0.404 (0.419) | 1.15 (1.64) | 0.0788 (0.878) | 0.332 (0.503) | numba-parallel | numpy |
| 1,000,000 | 4.21 (4.24) | 11.9 (12.4) | 0.704 (1.5) | 4.12 (4.29) | numba-parallel | numba-parallel |
| 10,000,000 | 50.9 (51.1) | 115 (116) | 6.91 (7.68) | 41.9 (42.5) | numba-parallel | numba-parallel |

### P7 `kalman`, batch sweep, float64, on aws-c8i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0239 (0.0245) | 0.00284 (7) | 0.00292 (10.7) | 0.00421 (0.261) | numba | numpy |
| 10 | 0.0405 (0.0406) | 0.0281 (7.05) | 0.0302 (10.7) | 0.0182 (0.277) | jax-cpu | numpy |
| 100 | 0.175 (0.175) | 0.282 (7.35) | 0.305 (11) | 0.157 (0.42) | jax-cpu | numpy |
| 1,000 | 1.58 (1.63) | 2.84 (9.89) | 2.99 (13.7) | 1.08 (1.33) | jax-cpu | jax-cpu |
| 10,000 | 15.2 (15.5) | 28.4 (35.4) | 30.8 (41.4) | 17.4 (17.6) | numpy | numpy |

### P7 `kalman`, batch sweep, float32, on aws-c8i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0261 (0.0266) | 0.00262 (7.11) | 0.00266 (10.7) | 0.00399 (0.262) | numba | numpy |
| 10 | 0.0429 (0.0433) | 0.0264 (7.05) | 0.0305 (10.7) | 0.0167 (0.277) | jax-cpu | numpy |
| 100 | 0.167 (0.167) | 0.261 (7.27) | 0.301 (10.9) | 0.129 (0.389) | jax-cpu | numpy |
| 1,000 | 1.41 (1.42) | 2.65 (9.65) | 3.03 (13.7) | 0.956 (1.23) | jax-cpu | jax-cpu |
| 10,000 | 13.8 (13.8) | 26.4 (33.3) | 29.5 (40.1) | 15.7 (15.9) | numpy | numpy |

### P7 `kalman`, state sweep, float64, on aws-c8i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0224 (0.0228) | 0.00211 (7.09) | 0.00216 (10.8) | 0.00154 (0.247) | jax-cpu | numpy |
| 5 | 0.0227 (0.0231) | 0.00235 (7.07) | 0.00237 (10.8) | 0.0029 (0.254) | numba | numpy |
| 10 | 0.0233 (0.0238) | 0.00284 (7.07) | 0.00291 (10.7) | 0.00409 (0.263) | numba | numpy |
| 20 | 0.0257 (0.0261) | 0.0046 (7.1) | 0.0045 (10.7) | 0.00597 (0.262) | numba-parallel | numpy |
| 50 | 0.0384 (0.0387) | 0.016 (7.05) | 0.0163 (10.8) | 0.0208 (0.28) | numba | numpy |
| 100 | 0.104 (0.108) | 0.0702 (7.13) | 0.0683 (10.8) | 0.0981 (0.366) | numba-parallel | numpy |
| 200 | 0.232 (0.232) | 0.199 (7.22) | 0.2 (10.9) | 0.487 (0.749) | numba | numpy |

### P7 `kalman`, state sweep, float32, on aws-c8i.8xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0257 (0.0262) | 0.00213 (7.01) | 0.00214 (10.7) | 0.00159 (0.246) | jax-cpu | numpy |
| 5 | 0.0264 (0.0267) | 0.00231 (7.02) | 0.00239 (10.6) | 0.00395 (0.27) | numba | numpy |
| 10 | 0.0267 (0.0272) | 0.00263 (7.04) | 0.00266 (10.7) | 0.00398 (0.26) | numba | numpy |
| 20 | 0.0275 (0.0278) | 0.00389 (7.05) | 0.00388 (10.7) | 0.00527 (0.264) | numba-parallel | numpy |
| 50 | 0.035 (0.0353) | 0.0113 (7.08) | 0.011 (10.7) | 0.0166 (0.277) | numba-parallel | numpy |
| 100 | 0.0823 (0.0821) | 0.0478 (7.11) | 0.0471 (10.7) | 0.0635 (0.326) | numba-parallel | numpy |
| 200 | 0.168 (0.172) | 0.153 (7.17) | 0.143 (10.8) | 0.26 (0.519) | numba-parallel | numpy |

### P1 `vfi`, grid sweep, float64, on aws-c8i.8xlarge (ynn-reduce-off)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 45.6 (45.4) | jax-cpu | jax-cpu |
| 1,500 | 103 (103) | jax-cpu | jax-cpu |

