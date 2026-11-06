### P1 `vfi`, grid sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 0.315 (0.319) | 0.173 (0.724) | 0.108 (0.864) | 0.0724 (0.103) | jax-cpu | jax-cpu |
| 200 | 1.38 (1.39) | 0.802 (1.35) | 0.187 (0.944) | 0.162 (0.188) | jax-cpu | jax-cpu |
| 400 | 5.81 (5.83) | 3.69 (4.25) | 0.45 (1.23) | 0.43 (0.465) | jax-cpu | jax-cpu |
| 700 | 18.1 (18.2) | 12 (12.7) | 1.28 (2.06) | 1.12 (1.14) | jax-cpu | jax-cpu |
| 1,000 | 36.6 (37.1) | 24.9 (25.7) | 2.68 (3.29) | 2.2 (2.2) | jax-cpu | jax-cpu |
| 1,500 | 85.1 (86.2) | 57.3 (58.2) | 6.52 (6.54) | 4.99 (5.02) | jax-cpu | jax-cpu |

### P3 `egm`, grid sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 200 | 0.236 (0.246) | 0.188 (1.65) | 0.443 (1.97) | 0.289 (0.353) | numba | numpy |
| 1,000 | 0.889 (0.892) | 0.882 (2.32) | 0.972 (2.46) | 1.35 (1.42) | numba | numpy |
| 5,000 | 4.19 (4.22) | 4.51 (5.85) | 3.81 (5.24) | 6.23 (6.3) | numba-parallel | numpy |
| 20,000 | 17.2 (17.3) | 18.2 (19.7) | 13.9 (15.4) | 27.7 (27.8) | numba-parallel | numba-parallel |
| 50,000 | 53.7 (53.9) | 46.1 (47.3) | 34.6 (36) | 80.5 (80.6) | numba-parallel | numba-parallel |

### P3 `egm`, grid sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 200 | 0.163 (0.165) | 0.138 (1.64) | 0.395 (1.95) | 0.174 (0.237) | numba | numpy |
| 1,000 | 0.643 (0.632) | 0.639 (2.12) | 0.702 (2.23) | 0.941 (1) | numba | numpy |
| 5,000 | 2.83 (2.8) | 3.13 (4.58) | 2.5 (4.03) | 4.4 (4.46) | numba-parallel | numpy |
| 20,000 | 11.1 (11.2) | 12.9 (14.3) | 8.3 (9.77) | 19 (19) | numba-parallel | numba-parallel |
| 50,000 | 28.1 (28.3) | 31.4 (32.8) | 20.2 (21.7) | 52.8 (52.9) | numba-parallel | numba-parallel |

- 200, `numpy`: 2,208 iterations, against NumPy's 2,192 in float64
- 200, `numba`: 2,208 iterations, against NumPy's 2,192 in float64
- 200, `numba-parallel`: 2,208 iterations, against NumPy's 2,192 in float64
- 200, `jax-cpu`: 2,208 iterations, against NumPy's 2,192 in float64
- 1,000, `numpy`: 2,206 iterations, against NumPy's 2,190 in float64
- 1,000, `numba`: 2,206 iterations, against NumPy's 2,190 in float64
- 1,000, `numba-parallel`: 2,206 iterations, against NumPy's 2,190 in float64
- 1,000, `jax-cpu`: 2,206 iterations, against NumPy's 2,190 in float64
- 5,000, `numpy`: 2,210 iterations, against NumPy's 2,190 in float64
- 5,000, `numba`: 2,210 iterations, against NumPy's 2,190 in float64
- 5,000, `numba-parallel`: 2,210 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-cpu`: 2,210 iterations, against NumPy's 2,190 in float64
- 20,000, `numpy`: 2,210 iterations, against NumPy's 2,190 in float64
- 20,000, `numba`: 2,210 iterations, against NumPy's 2,190 in float64
- 20,000, `numba-parallel`: 2,210 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-cpu`: 2,210 iterations, against NumPy's 2,190 in float64
- 50,000, `numpy`: 2,210 iterations, against NumPy's 2,190 in float64
- 50,000, `numba`: 2,210 iterations, against NumPy's 2,190 in float64
- 50,000, `numba-parallel`: 2,210 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-cpu`: 2,210 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00343 (0.00347) | 0.0078 (0.245) | 0.000623 (0.372) | 0.00961 (0.102) | numba-parallel | numpy |
| 10,000 | 0.0283 (0.0285) | 0.0764 (0.311) | 0.00506 (0.374) | 0.101 (0.191) | numba-parallel | numpy |
| 100,000 | 0.269 (0.27) | 0.769 (1) | 0.0485 (0.414) | 0.957 (1.05) | numba-parallel | numpy |
| 1,000,000 | 2.7 (2.69) | 7.56 (7.78) | 0.481 (0.833) | 9.51 (9.66) | numba-parallel | numba-parallel |
| 10,000,000 | 26.7 (26.7) | 75.6 (75.7) | 5.41 (5.11) | 30.4 (30.6) | numba-parallel | numba-parallel |

### P4 `wealth`, households sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00258 (0.00263) | 0.00515 (0.244) | 0.00053 (0.374) | 0.00357 (0.0891) | numba-parallel | numpy |
| 10,000 | 0.0166 (0.0167) | 0.0528 (0.291) | 0.00403 (0.378) | 0.0359 (0.118) | numba-parallel | numpy |
| 100,000 | 0.156 (0.155) | 0.519 (0.78) | 0.0397 (0.409) | 0.362 (0.445) | numba-parallel | numpy |
| 1,000,000 | 1.53 (1.54) | 5.28 (5.5) | 0.379 (0.747) | 3.63 (3.72) | numba-parallel | numba-parallel |
| 10,000,000 | 15.2 (15.4) | 53 (53.5) | 4.09 (4.01) | 21.9 (22.4) | numba-parallel | numba-parallel |

### P7 `kalman`, batch sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0159 (0.0162) | 0.00159 (2.61) | 0.00193 (3.57) | 0.00182 (0.141) | numba | numpy |
| 10 | 0.0302 (0.0293) | 0.0152 (2.61) | 0.00279 (3.56) | 0.00667 (0.135) | numba-parallel | numpy |
| 100 | 0.135 (0.136) | 0.153 (2.73) | 0.0175 (3.55) | 0.0754 (0.204) | numba-parallel | numpy |
| 1,000 | 1.15 (1.16) | 1.49 (4.09) | 0.153 (3.64) | 0.428 (0.56) | numba-parallel | jax-cpu |
| 10,000 | 11.1 (11.2) | 14.8 (17.7) | 1.57 (4.9) | 2.42 (2.54) | numba-parallel | jax-cpu |

### P7 `kalman`, batch sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0147 (0.0151) | 0.00178 (2.66) | 0.00215 (3.7) | 0.00499 (0.143) | numba | numpy |
| 10 | 0.0264 (0.0271) | 0.0176 (2.67) | 0.00312 (3.7) | 0.0163 (0.15) | numba-parallel | numpy |
| 100 | 0.135 (0.136) | 0.177 (2.82) | 0.0169 (3.68) | 0.096 (0.234) | numba-parallel | numpy |
| 1,000 | 1.18 (1.19) | 1.74 (4.39) | 0.152 (3.74) | 0.537 (0.673) | numba-parallel | jax-cpu |
| 10,000 | 11.4 (11.6) | 17.4 (20.7) | 1.58 (5.01) | 2.8 (2.95) | numba-parallel | jax-cpu |

### P7 `kalman`, state sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0124 (0.0128) | 0.0011 (2.61) | 0.00145 (3.57) | 0.000648 (0.128) | jax-cpu | numpy |
| 5 | 0.0124 (0.0127) | 0.00127 (2.59) | 0.0016 (3.55) | 0.00129 (0.127) | numba | numpy |
| 10 | 0.0157 (0.016) | 0.00153 (2.58) | 0.00182 (3.55) | 0.00176 (0.141) | numba | numpy |
| 20 | 0.0174 (0.018) | 0.0033 (2.57) | 0.00363 (3.52) | 0.00288 (0.141) | jax-cpu | numpy |
| 50 | 0.0225 (0.0231) | 0.00872 (2.57) | 0.00909 (3.52) | 0.0167 (0.154) | numba | numpy |
| 100 | 0.0415 (0.0416) | 0.0299 (2.57) | 0.0299 (3.53) | 0.115 (0.255) | numba-parallel | numpy |
| 200 | 0.126 (0.128) | 0.119 (2.65) | 0.117 (3.59) | 0.555 (0.682) | numba-parallel | numpy |

### P7 `kalman`, state sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0138 (0.014) | 0.00111 (2.58) | 0.00137 (3.6) | 0.000645 (0.123) | jax-cpu | numpy |
| 5 | 0.0136 (0.014) | 0.00126 (2.57) | 0.00151 (3.58) | 0.00128 (0.122) | numba | numpy |
| 10 | 0.0144 (0.0146) | 0.00181 (2.58) | 0.00203 (3.58) | 0.005 (0.137) | numba | numpy |
| 20 | 0.0182 (0.0186) | 0.00349 (2.57) | 0.00387 (3.56) | 0.00646 (0.137) | numba | numpy |
| 50 | 0.0236 (0.0241) | 0.0093 (2.56) | 0.0096 (3.57) | 0.0124 (0.146) | numba | numpy |
| 100 | 0.0334 (0.0344) | 0.0236 (2.58) | 0.0239 (3.58) | 0.0317 (0.165) | numba | numpy |
| 200 | 0.0684 (0.0715) | 0.0753 (2.63) | 0.0754 (3.64) | 0.143 (0.277) | numpy | numpy |

