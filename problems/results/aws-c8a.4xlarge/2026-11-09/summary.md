### P1 `vfi`, grid sweep, float64, on aws-c8a.4xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 0.528 (0.54) | 0.208 (1.35) | 0.071 (1.81) | 0.0429 (0.0874) | jax-cpu | jax-cpu |
| 200 | 1.89 (1.9) | 1.2 (2.35) | 0.23 (1.95) | 0.0897 (0.137) | jax-cpu | jax-cpu |
| 400 | 11.6 (11.7) | 5.61 (6.74) | 3.31 (3.76) | 0.578 (0.625) | jax-cpu | jax-cpu |
| 700 | 36.6 (36.5) | 22.9 (24.1) | 11.7 (13.2) | 2.16 (2.2) | jax-cpu | jax-cpu |
| 1,000 | 86.3 (81.8) | 58.2 (52.7) | 19.9 (22) | 4.63 (4.64) | jax-cpu | jax-cpu |
| 1,500 | 183 (184) | 118 (114) | 28 (28.6) | 10.3 (10.3) | jax-cpu | jax-cpu |

### P3 `egm`, grid sweep, float64, on aws-c8a.4xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 200 | 0.188 (0.199) | 0.267 (3.37) | 0.373 (3.6) | 0.355 (0.476) | numpy | numpy |
| 1,000 | 0.633 (0.63) | 1.32 (4.43) | 1.45 (4.71) | 1.7 (1.82) | numpy | numpy |
| 5,000 | 3.25 (3.31) | 7.19 (10.3) | 6.88 (10.2) | 8.52 (8.67) | numpy | numpy |
| 20,000 | 13.8 (13.9) | 29.7 (32.8) | 30.6 (33.4) | 35.1 (35.1) | numpy | numpy |
| 50,000 | 37.2 (36.3) | 78.7 (82.4) | 81.8 (80.1) | 132 (132) | numpy | numpy |

### P3 `egm`, grid sweep, float32, on aws-c8a.4xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 200 | 0.181 (0.186) | 0.163 (3.32) | 0.267 (3.71) | 0.203 (0.338) | numba | numpy |
| 1,000 | 0.498 (0.502) | 0.739 (3.88) | 0.968 (4.38) | 1.07 (1.1) | numpy | numpy |
| 5,000 | 2.74 (2.76) | 4.17 (7.33) | 4.62 (8) | 5.73 (5.77) | numpy | numpy |
| 20,000 | 10.5 (10.5) | 17.2 (20.5) | 18.2 (21.7) | 25.7 (26.1) | numpy | numpy |
| 50,000 | 26.6 (26.7) | 44.6 (48.1) | 44.7 (49.7) | 75.9 (75.4) | numpy | numpy |

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

### P4 `wealth`, households sweep, float64, on aws-c8a.4xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00493 (0.00519) | 0.0138 (0.433) | 0.00162 (0.67) | 0.00586 (0.175) | numba-parallel | numpy |
| 10,000 | 0.0379 (0.0379) | 0.136 (0.555) | 0.013 (0.673) | 0.0599 (0.223) | numba-parallel | numpy |
| 100,000 | 0.387 (0.397) | 1.36 (1.77) | 0.0816 (0.766) | 0.608 (0.771) | numba-parallel | numpy |
| 1,000,000 | 3.91 (3.89) | 13.2 (14) | 0.838 (1.5) | 6.37 (6.51) | numba-parallel | numba-parallel |
| 10,000,000 | 46.8 (46.8) | 131 (132) | 8.41 (9.19) | 29.1 (29.3) | numba-parallel | numba-parallel |

### P4 `wealth`, households sweep, float32, on aws-c8a.4xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00466 (0.00478) | 0.012 (0.425) | 0.00158 (0.681) | 0.0017 (0.154) | numba-parallel | numpy |
| 10,000 | 0.0333 (0.0331) | 0.122 (0.535) | 0.0124 (0.688) | 0.017 (0.166) | numba-parallel | numpy |
| 100,000 | 0.327 (0.339) | 1.21 (1.63) | 0.0842 (0.768) | 0.188 (0.334) | numba-parallel | jax-cpu |
| 1,000,000 | 3.3 (3.33) | 12.1 (12.5) | 0.799 (1.48) | 2.16 (2.27) | numba-parallel | numba-parallel |
| 10,000,000 | 38.2 (38.2) | 122 (122) | 7.94 (8.67) | 19.6 (19.7) | numba-parallel | numba-parallel |

### P7 `kalman`, batch sweep, float64, on aws-c8a.4xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0196 (0.0201) | 0.00234 (6.2) | 0.00253 (9.57) | 0.00337 (0.24) | numba | numpy |
| 10 | 0.0311 (0.0313) | 0.0238 (6.21) | 0.0299 (9.58) | 0.013 (0.255) | jax-cpu | numpy |
| 100 | 0.129 (0.131) | 0.235 (6.43) | 0.299 (9.83) | 0.127 (0.368) | jax-cpu | numpy |
| 1,000 | 1.11 (1.15) | 2.36 (8.54) | 2.98 (12.5) | 1 (1.25) | jax-cpu | numpy |
| 10,000 | 11.2 (11.5) | 23.6 (29.8) | 29.5 (39.1) | 11.4 (11.7) | numpy | numpy |

### P7 `kalman`, batch sweep, float32, on aws-c8a.4xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0222 (0.0222) | 0.00225 (6.17) | 0.0023 (9.53) | 0.00338 (0.242) | numba | numpy |
| 10 | 0.0325 (0.0325) | 0.0219 (6.18) | 0.0289 (9.58) | 0.0129 (0.253) | jax-cpu | numpy |
| 100 | 0.126 (0.126) | 0.223 (6.39) | 0.294 (9.89) | 0.115 (0.341) | jax-cpu | numpy |
| 1,000 | 1.06 (1.08) | 2.21 (8.37) | 2.98 (12.5) | 0.841 (1.08) | jax-cpu | jax-cpu |
| 10,000 | 10.6 (10.6) | 22.3 (28.4) | 29.8 (39.3) | 11.2 (11.4) | numpy | numpy |

### P7 `kalman`, state sweep, float64, on aws-c8a.4xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0194 (0.0199) | 0.00185 (6.17) | 0.00192 (9.56) | 0.00139 (0.231) | jax-cpu | numpy |
| 5 | 0.0197 (0.02) | 0.00206 (6.17) | 0.00214 (9.54) | 0.00259 (0.232) | numba | numpy |
| 10 | 0.0204 (0.0209) | 0.00232 (6.16) | 0.00255 (9.54) | 0.00341 (0.24) | numba | numpy |
| 20 | 0.0212 (0.0216) | 0.00349 (6.24) | 0.00354 (9.61) | 0.00442 (0.238) | numba | numpy |
| 50 | 0.0281 (0.0309) | 0.0107 (6.25) | 0.0103 (9.51) | 0.0146 (0.256) | numba-parallel | numpy |
| 100 | 0.0784 (0.0815) | 0.0448 (6.25) | 0.0448 (9.65) | 0.0631 (0.313) | numba | numpy |
| 200 | 0.349 (0.35) | 0.25 (6.44) | 0.275 (9.86) | 0.343 (0.564) | numba | numpy |

### P7 `kalman`, state sweep, float32, on aws-c8a.4xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0212 (0.0214) | 0.00182 (6.19) | 0.00192 (9.56) | 0.00137 (0.23) | jax-cpu | numpy |
| 5 | 0.0216 (0.0225) | 0.0021 (6.18) | 0.0021 (9.53) | 0.00343 (0.247) | numba | numpy |
| 10 | 0.0218 (0.0233) | 0.0022 (6.17) | 0.00233 (9.55) | 0.00348 (0.24) | numba | numpy |
| 20 | 0.0225 (0.024) | 0.00304 (6.24) | 0.00332 (9.54) | 0.00418 (0.241) | numba | numpy |
| 50 | 0.0287 (0.0279) | 0.00825 (6.2) | 0.00814 (9.55) | 0.011 (0.248) | numba-parallel | numpy |
| 100 | 0.0573 (0.0582) | 0.0276 (6.21) | 0.0279 (9.57) | 0.043 (0.289) | numba | numpy |
| 200 | 0.272 (0.271) | 0.207 (6.39) | 0.219 (9.82) | 0.169 (0.402) | jax-cpu | numpy |

### P1 `vfi`, grid sweep, float64, on aws-c8a.4xlarge (ynn-reduce-off)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 20.4 (20.4) | jax-cpu | jax-cpu |
| 1,500 | 45.7 (45.6) | jax-cpu | jax-cpu |

