### P2 `hpi`, grid sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 100 | 0.188 (0.206) | 0.132 (6.96) | 0.137 (7.64) | 0.119 (0.624) | 0.0757 (1.31) | jax-gpu | numpy |
| 200 | 0.613 (0.566) | 0.23 (7) | 0.225 (7.74) | 0.303 (0.829) | 0.0694 (1.29) | jax-gpu | numpy |
| 400 | 1.34 (1.33) | 0.782 (7.54) | 0.551 (8.06) | 0.655 (1.09) | 0.0757 (1.31) | jax-gpu | jax-cpu |
| 700 | 3.02 (3.16) | 1.96 (8.7) | 1.16 (8.73) | 1.16 (1.61) | 0.0826 (1.36) | jax-gpu | jax-gpu |
| 1,000 | 5.46 (6.71) | 3.4 (10.2) | 1.79 (9.36) | 1.68 (2.05) | 0.0993 (1.41) | jax-gpu | jax-gpu |
| 1,500 | 10.9 (11) | 7.23 (14) | 3 (10.7) | 2.37 (2.68) | 0.148 (1.49) | jax-gpu | jax-gpu |

### P2 `hpi`, income sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 25 | 0.0743 (0.0775) | 0.0344 (6.78) | 0.0415 (7.63) | 0.0347 (0.569) | 0.0471 (1.3) | numba | numpy |
| 50 | 0.118 (0.123) | 0.0719 (6.84) | 0.0729 (7.84) | 0.0662 (0.573) | 0.0469 (1.23) | jax-gpu | numpy |
| 100 | 0.407 (0.41) | 0.167 (6.93) | 0.164 (7.9) | 0.175 (0.694) | 0.0785 (1.27) | jax-gpu | numpy |
| 200 | 0.869 (0.912) | 0.513 (7.23) | 0.558 (8.09) | 0.494 (1.08) | 0.0683 (1.37) | jax-gpu | numpy |
| 400 | 1.69 (1.78) | 0.924 (7.92) | 0.881 (8.43) | 0.91 (1.44) | 0.1 (1.36) | jax-gpu | jax-gpu |

### P5 `stationary`, dense sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 100 | 0.00106 (0.00605) | 0.000617 (6.84) | 0.0056 (7.02) | 0.00124 (0.354) | 0.00202 (0.984) | numba | numpy |
| 200 | 0.00369 (0.00698) | 0.00514 (6.36) | 0.00993 (7.14) | 0.00433 (0.368) | 0.00384 (0.919) | numpy | numpy |
| 500 | 0.0633 (0.076) | 0.0509 (5.84) | 0.0527 (6.62) | 0.11 (0.517) | 0.0113 (0.951) | jax-gpu | numpy |
| 1,000 | 0.322 (0.311) | 0.334 (6.09) | 0.321 (6.79) | 0.265 (0.548) | 0.0382 (0.938) | jax-gpu | numpy |
| 2,000 | 1.42 (1.46) | 2.25 (8.25) | 2.19 (8.72) | 0.983 (1.13) | 0.114 (1.06) | jax-gpu | jax-gpu |
| 5,000 | 8.92 (8.54) | 8.83 (14.7) | 8.42 (15.1) | 6.01 (6.36) | 1.33 (2.31) | jax-gpu | jax-gpu |

### P5 `stationary`, dense sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 100 | 0.000667 (0.02) | 0.000376 (6.19) | 0.0036 (7.01) | 0.000772 (0.355) | 0.00166 (0.905) | numba | numpy |
| 200 | 0.00233 (0.0037) | 0.00275 (6.22) | 0.00664 (7) | 0.00306 (0.346) | 0.00248 (0.868) | numpy | numpy |
| 500 | 0.0243 (0.0296) | 0.0279 (5.81) | 0.0302 (6.55) | 0.0181 (0.371) | 0.0045 (0.968) | jax-gpu | numpy |
| 1,000 | 0.214 (0.269) | 0.192 (6.14) | 0.187 (6.67) | 0.219 (0.531) | 0.00922 (0.908) | jax-gpu | numpy |
| 2,000 | 0.79 (0.763) | 1.3 (7.06) | 1.28 (7.8) | 0.591 (0.781) | 0.0232 (0.916) | jax-gpu | numpy |
| 5,000 | 4.95 (4.65) | 18.2 (24.3) | 17.9 (25.2) | 2.81 (2.74) | 0.0971 (1.04) | jax-gpu | jax-gpu |

- 1,000, `jax-cpu`: answer differs from NumPy's in float64 (relative difference 0.002)
- 5,000, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.014)

### P5 `stationary`, sparse sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00649 (0.0213) | 0.00319 (5.1) | 0.0135 (7.45) | 0.00542 (0.429) | 0.00683 (1.06) | numba | numpy |
| 10,000 | 0.0401 (0.0416) | 0.034 (4.82) | 0.0196 (6.81) | 0.122 (0.522) | 0.00887 (1.07) | jax-gpu | numpy |
| 100,000 | 0.574 (0.59) | 0.368 (5.32) | 0.0703 (6.95) | 1.2 (1.53) | 0.0204 (1.03) | jax-gpu | numpy |
| 1,000,000 | 5.26 (5.28) | 3.92 (8.67) | 2.39 (9.08) | 12.3 (13.9) | 0.146 (1.18) | jax-gpu | jax-gpu |
| 10,000,000 | 59.6 (74) | 41.6 (46.9) | 19.6 (32.7) | 97.1 (104) | 1.54 (2.58) | jax-gpu | jax-gpu |

### P5 `stationary`, sparse sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00381 (0.00447) | 0.00197 (4.87) | 0.00782 (7) | 0.00193 (0.446) | 0.00379 (1.07) | jax-cpu | numpy |
| 10,000 | 0.0244 (0.0264) | 0.022 (5.05) | 0.0117 (7.04) | 0.0289 (0.459) | 0.00502 (0.986) | jax-gpu | numpy |
| 100,000 | 0.244 (0.262) | 0.238 (5.25) | 0.043 (7.08) | 0.537 (0.907) | 0.00834 (1) | jax-gpu | numpy |
| 1,000,000 | 2.65 (2.67) | 2.52 (7.27) | 0.98 (7.73) | 6.33 (6.85) | 0.0557 (1.1) | jax-gpu | jax-gpu |
| 10,000,000 | 30.1 (30.3) | 27 (31.8) | 11.6 (18.2) | 52.5 (55.7) | 0.542 (1.59) | jax-gpu | jax-gpu |

- NumPy's iterations in float32, against its float64 count: 157 against 266 at 1,000; 165 against 274 at 10,000; 174 against 283 at 100,000; 182 against 291 at 1,000,000; 192 against 299 at 10,000,000
- 10,000,000, `jax-gpu`: 195 iterations, against NumPy's 192 in float32

### P6 `time_iter`, grid sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 120 | 0.421 (0.423) | 0.235 (4.01) | 0.00511 (4.31) | 0.177 (0.853) | 0.0244 (1.44) | numba-parallel | numpy |
| 1,000 | 4.07 (4.11) | 2.43 (6.1) | 0.0442 (4.35) | 1.88 (2.65) | 0.0436 (1.58) | jax-gpu | jax-gpu |
| 5,000 | 22.9 (23) | 13.6 (17.3) | 0.249 (4.59) | 13.6 (13.7) | 0.222 (1.7) | jax-gpu | jax-gpu |
| 20,000 | 103 (103) | 61.7 (65.4) | 1.24 (5.59) | 57.3 (57.6) | 1 (2.48) | jax-gpu | jax-gpu |
| 50,000 | 275 (275) | 170 (173) | 3.39 (7.64) | 85.3 (78.6) | 2.6 (4.06) | jax-gpu | jax-gpu |

### P6 `time_iter`, grid sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 120 | 0.487 (0.484) | 0.238 (4.11) | 0.00726 (4.45) | 0.126 (0.831) | 0.0252 (1.42) | numba-parallel | numpy |
| 1,000 | 4.8 (4.87) | 2.49 (6.18) | 0.0481 (4.33) | 1.2 (1.88) | 0.0355 (1.47) | jax-gpu | jax-gpu |
| 5,000 | 28.5 (28.6) | 14.5 (18.2) | 0.294 (4.66) | 10.4 (11) | 0.181 (1.61) | jax-gpu | jax-gpu |
| 20,000 | 151 (151) | 65 (68.7) | 1.26 (5.77) | 41.4 (44.1) | 0.866 (2.32) | jax-gpu | jax-gpu |
| 50,000 | timeout | 176 (180) | 3.34 (7.57) | 111 (119) | 2 (3.44) | jax-gpu | jax-gpu |

### P6 `time_iter`, shocks sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 250 | 0.418 (0.421) | 0.237 (3.91) | 0.018 (4.22) | 0.177 (0.832) | 0.0241 (1.42) | numba-parallel | numpy |
| 1,000 | 0.724 (0.725) | 0.94 (4.63) | 0.0404 (4.28) | 0.602 (1.25) | 0.0267 (1.43) | jax-gpu | numpy |
| 5,000 | 2.35 (2.35) | 4.6 (8.28) | 0.101 (4.34) | 3.63 (4.35) | 0.0654 (1.47) | jax-gpu | jax-gpu |
| 20,000 | 8.49 (8.5) | 18.2 (21.9) | 0.404 (4.7) | 14.2 (14.9) | 0.273 (1.71) | jax-gpu | jax-gpu |
| 100,000 | 55.8 (55.8) | 94.4 (98.2) | 2.08 (6.31) | 36.6 (47.6) | 1.27 (2.74) | jax-gpu | jax-gpu |

### P6 `time_iter`, shocks sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 250 | 0.485 (0.484) | 0.237 (4.12) | 0.0219 (4.48) | 0.124 (0.816) | 0.0252 (1.43) | numba-parallel | numpy |
| 1,000 | 0.811 (0.811) | 0.943 (4.83) | 0.0188 (4.26) | 0.452 (1.09) | 0.0242 (1.42) | numba-parallel | numpy |
| 5,000 | 2.44 (2.44) | 4.67 (8.37) | 0.104 (4.35) | 2.56 (3.22) | 0.0472 (1.5) | jax-gpu | jax-gpu |
| 20,000 | 8.58 (8.59) | 18.7 (22.4) | 0.401 (4.96) | 9.3 (10.1) | 0.204 (1.58) | jax-gpu | jax-gpu |
| 100,000 | 59.4 (59.4) | 93.9 (97.5) | 1.98 (6.43) | 48.4 (49) | 1.01 (2.44) | jax-gpu | jax-gpu |

### P8 `mle`, obs sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 10,000 | 0.00318 (0.0179) | 0.00329 (7.46) | 0.0132 (7.79) | 0.00425 (0.601) | 0.0031 (1.14) | jax-gpu | numpy |
| 100,000 | 0.0416 (0.0427) | 0.0397 (7.79) | 0.0357 (7.91) | 0.0515 (0.608) | 0.00914 (1.2) | jax-gpu | numpy |
| 500,000 | 0.328 (0.336) | 0.428 (7.95) | 0.563 (8.59) | 0.349 (0.886) | 0.0596 (1.67) | jax-gpu | numpy |
| 1,000,000 | 0.459 (0.471) | 0.577 (7.96) | 0.579 (8.34) | 0.502 (1.07) | 0.0792 (2.25) | jax-gpu | numpy |
| 3,000,000 | 2 (2.02) | 2.99 (10.5) | 2.11 (10.1) | 2.02 (2.68) | 0.29 (4.77) | jax-gpu | numpy |
| 10,000,000 | 6.1 (6.18) | 8.64 (16.2) | 6.25 (13.9) | 5.85 (6.42) | 0.901 (13.4) | jax-gpu | numpy |

### P8 `mle`, obs sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 10,000 | 0.00248 (0.0182) | 0.0026 (7.78) | 0.013 (7.83) | 0.00379 (0.581) | 0.00215 (1.11) | jax-gpu | numpy |
| 100,000 | 0.024 (0.0251) | 0.0309 (7.83) | 0.0344 (7.77) | 0.046 (0.594) | 0.0027 (1.1) | jax-gpu | numpy |
| 500,000 | 0.212 (0.23) | 0.301 (7.74) | 0.518 (8.26) | 0.28 (0.816) | 0.0126 (1.11) | jax-gpu | numpy |
| 1,000,000 | 0.301 (0.304) | 0.419 (7.8) | 0.449 (8.28) | 0.403 (0.922) | 0.0136 (1.11) | jax-gpu | numpy |
| 3,000,000 | 1.3 (1.31) | 2.02 (9.43) | 1.58 (9.44) | 1.55 (2.04) | 0.0319 (1.18) | jax-gpu | jax-gpu |
| 10,000,000 | 3.98 (4) | 5.87 (13.3) | 4.09 (11.9) | 4.84 (5.41) | 0.0934 (1.6) | jax-gpu | jax-gpu |

- 3,000,000, `jax-cpu`: 18 iterations, against NumPy's 17 in float32
- 10,000,000, `jax-cpu`: 19 iterations, against NumPy's 16 in float32

### P8 `mle`, params sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 3 | 0.0243 (0.0251) | 0.0231 (7.43) | 0.0193 (7.83) | 0.0328 (0.583) | 0.00608 (1.19) | jax-gpu | numpy |
| 10 | 0.0617 (0.0658) | 0.054 (7.5) | 0.202 (8.03) | 0.0314 (0.648) | 0.018 (1.18) | jax-gpu | numpy |
| 30 | 0.303 (0.301) | 0.299 (7.74) | 0.298 (8.06) | 0.0985 (0.71) | 0.0156 (1.28) | jax-gpu | numpy |
| 100 | 0.824 (0.788) | 1.12 (8.57) | 1.06 (8.9) | 0.418 (1.02) | 0.0982 (1.35) | jax-gpu | numpy |
| 300 | 3.6 (3.25) | 4.53 (12.4) | 4.37 (11.9) | 2.61 (3.81) | 0.689 (2.17) | jax-gpu | jax-gpu |
| 1,000 | 22.6 (21.1) | 28.2 (36.4) | 26.9 (34.2) | 13.7 (13.3) | 11.6 (16.6) | jax-gpu | jax-cpu |

### P8 `mle`, params sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---:|---|---|
| 3 | 0.0135 (0.0148) | 0.018 (7.56) | 0.029 (7.92) | 0.0269 (0.568) | 0.0018 (1.12) | jax-gpu | numpy |
| 10 | 0.0366 (0.0396) | 0.058 (7.54) | 0.201 (8.06) | 0.0202 (0.661) | 0.0018 (1.01) | jax-gpu | numpy |
| 30 | 0.159 (0.16) | 0.239 (7.75) | 0.238 (8.09) | 0.044 (0.698) | 0.00295 (1.1) | jax-gpu | numpy |
| 100 | 0.445 (0.447) | 0.658 (8.18) | 0.588 (8.47) | 0.164 (0.804) | 0.0106 (1.1) | jax-gpu | numpy |
| 300 | 1.83 (1.69) | 2.41 (10.1) | 2.47 (10.3) | 0.74 (1.47) | 0.0469 (1.26) | jax-gpu | jax-gpu |
| 1,000 | 11.6 (13.9) | 14.8 (22.6) | 13.5 (21.1) | 5.64 (6.67) | 0.507 (2.16) | jax-gpu | jax-gpu |

### P8 `mle_hand`, obs sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.00426 (0.522) | 0.00271 (1.04) | jax-gpu | jax-cpu |
| 100,000 | 0.0382 (0.548) | 0.00729 (1.22) | jax-gpu | jax-cpu |
| 500,000 | 0.31 (0.791) | 0.0437 (2.14) | jax-gpu | jax-cpu |
| 1,000,000 | 0.464 (0.947) | 0.0575 (3.26) | jax-gpu | jax-cpu |
| 3,000,000 | 1.91 (2.32) | 0.28 (7.87) | jax-gpu | jax-cpu |
| 10,000,000 | 5.53 (6.55) | 0.869 (23.6) | jax-gpu | jax-cpu |

### P8 `mle_hand`, obs sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.00256 (0.53) | 0.00204 (1) | jax-gpu | jax-cpu |
| 100,000 | 0.0385 (0.563) | 0.00218 (1.01) | jax-gpu | jax-cpu |
| 500,000 | 0.273 (0.812) | 0.00696 (1.01) | jax-gpu | jax-cpu |
| 1,000,000 | 0.365 (0.88) | 0.0086 (1.05) | jax-gpu | jax-cpu |
| 3,000,000 | 1.43 (1.96) | 0.0296 (1.01) | jax-gpu | jax-gpu |
| 10,000,000 | 4.74 (5.42) | 0.0569 (1.09) | jax-gpu | jax-gpu |

- 3,000,000, `jax-cpu`: 18 iterations, against NumPy's 17 in float32
- 10,000,000, `jax-cpu`: 19 iterations, against NumPy's 16 in float32

### P8 `mle_hand`, params sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.0235 (0.534) | 0.00454 (1.17) | jax-gpu | jax-cpu |
| 10 | 0.0209 (0.62) | 0.0125 (1.18) | jax-gpu | jax-cpu |
| 30 | 0.0739 (0.637) | 0.0072 (1.3) | jax-gpu | jax-cpu |
| 100 | 0.398 (0.95) | 0.0567 (1.28) | jax-gpu | jax-cpu |
| 300 | 2.58 (3.76) | 0.364 (1.74) | jax-gpu | jax-gpu |
| 1,000 | 12.9 (14) | 5.96 (9.5) | jax-gpu | jax-gpu |

### P8 `mle_hand`, params sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.0189 (0.511) | 0.00162 (0.964) | jax-gpu | jax-cpu |
| 10 | 0.018 (0.581) | 0.00167 (0.964) | jax-gpu | jax-cpu |
| 30 | 0.0434 (0.624) | 0.0024 (1.03) | jax-gpu | jax-cpu |
| 100 | 0.148 (0.738) | 0.00777 (0.994) | jax-gpu | jax-cpu |
| 300 | 0.733 (1.43) | 0.0327 (1.08) | jax-gpu | jax-gpu |
| 1,000 | 5.98 (6.61) | 0.31 (1.64) | jax-gpu | jax-gpu |

### P8 `mle_onepass`, obs sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.00293 (9.28) | 0.0152 (13.9) | numba | numba |
| 100,000 | 0.035 (9.74) | 0.0343 (13.8) | numba-parallel | numba |
| 500,000 | 0.247 (9.55) | 0.0307 (13.7) | numba-parallel | numba |
| 1,000,000 | 0.35 (9.91) | 0.0291 (13.3) | numba-parallel | numba |
| 3,000,000 | 1.49 (11.1) | 0.0432 (13.7) | numba-parallel | numba |
| 10,000,000 | 4.66 (14.1) | 0.205 (14) | numba-parallel | numba-parallel |

### P8 `mle_onepass`, obs sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.00244 (9.63) | 0.0139 (13.8) | numba | numba |
| 100,000 | 0.029 (9.57) | 0.0178 (13.7) | numba-parallel | numba |
| 500,000 | 0.206 (9.47) | 0.0279 (13.6) | numba-parallel | numba |
| 1,000,000 | 0.29 (9.65) | 0.0283 (13.3) | numba-parallel | numba |
| 3,000,000 | 1.24 (10.7) | 0.047 (13.7) | numba-parallel | numba |
| 10,000,000 | 3.86 (13.4) | 0.145 (13.8) | numba-parallel | numba |

### P8 `mle_onepass`, params sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.0204 (9.34) | 0.0114 (13.7) | numba-parallel | numba |
| 10 | 0.0505 (9.41) | 0.0121 (13.8) | numba-parallel | numba |
| 30 | 0.188 (9.8) | 0.0169 (13.7) | numba-parallel | numba |
| 100 | 1.65 (10.9) | 0.109 (13.7) | numba-parallel | numba |
| 300 | 15.9 (25.5) | 0.853 (14.7) | numba-parallel | numba-parallel |

### P8 `mle_onepass`, params sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.0169 (9.31) | 0.0101 (13.6) | numba-parallel | numba |
| 10 | 0.0427 (9.61) | 0.0144 (13.6) | numba-parallel | numba |
| 30 | 0.151 (9.6) | 0.0119 (13.8) | numba-parallel | numba |
| 100 | 1.04 (10.6) | 0.0683 (13.7) | numba-parallel | numba |
| 300 | 8.37 (17.6) | 0.523 (14.3) | numba-parallel | numba-parallel |

### P8 `mle_blocked`, obs sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.00358 (11.7) | 0.0152 (17.2) | numba | numba |
| 100,000 | 0.0419 (12.3) | 0.041 (17.2) | numba-parallel | numba |
| 500,000 | 0.302 (12.3) | 0.255 (17.6) | numba-parallel | numba |
| 1,000,000 | 0.427 (12.4) | 0.351 (16.8) | numba-parallel | numba |
| 3,000,000 | 1.79 (13.7) | 1.51 (18.6) | numba-parallel | numba |
| 10,000,000 | 5.66 (17.7) | 4.7 (21.8) | numba-parallel | numba |

### P8 `mle_blocked`, obs sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.00248 (11.9) | 0.0133 (17.2) | numba | numba |
| 100,000 | 0.0296 (12.1) | 0.0121 (16.9) | numba-parallel | numba |
| 500,000 | 0.21 (12.3) | 0.0752 (17) | numba-parallel | numba |
| 1,000,000 | 0.298 (12) | 0.105 (16.6) | numba-parallel | numba |
| 3,000,000 | 1.26 (13.1) | 0.412 (17.3) | numba-parallel | numba |
| 10,000,000 | 3.97 (15.7) | 1.31 (18.4) | numba-parallel | numba |

### P8 `mle_blocked`, params sweep, float64, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.0244 (11.9) | 0.028 (17) | numba | numba |
| 10 | 0.0437 (11.9) | 0.0292 (17) | numba-parallel | numba |
| 30 | 0.134 (12.3) | 0.0296 (16.8) | numba-parallel | numba |
| 100 | 0.995 (12.9) | 0.691 (17.7) | numba-parallel | numba |
| 300 | 6 (18) | 11.8 (32) | numba | numba |
| 1,000 | 50.7 (62.8) | 43.5 (58.3) | numba-parallel | numba-parallel |

### P8 `mle_blocked`, params sweep, float32, on qe-wharton1-gpu

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.0173 (11.9) | 0.0139 (16.9) | numba-parallel | numba |
| 10 | 0.0297 (11.9) | 0.014 (16.9) | numba-parallel | numba |
| 30 | 0.234 (12) | 0.139 (17) | numba-parallel | numba |
| 100 | 0.464 (12.3) | 0.295 (17.3) | numba-parallel | numba |
| 300 | 2.09 (14.4) | 6.28 (23.1) | numba | numba |
| 1,000 | 28.8 (41.2) | 24.3 (41.2) | numba-parallel | numba-parallel |

### P8 `mle_blocked`, params sweep, float64, on qe-wharton1-gpu (openblas-1-thread)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.0238 (11.5) | 0.0185 (16.3) | numba-parallel | numba |
| 10 | 0.0432 (11.3) | 0.0198 (16.4) | numba-parallel | numba |
| 30 | 0.134 (11.4) | 0.0242 (16.3) | numba-parallel | numba |
| 100 | 0.974 (12.5) | 0.111 (16.4) | numba-parallel | numba |
| 300 | 8.85 (20.3) | 1.03 (17.3) | numba-parallel | numba-parallel |
| 1,000 | 162 (173) | 20.3 (37.3) | numba-parallel | numba-parallel |

### P8 `mle_blocked`, params sweep, float32, on qe-wharton1-gpu (openblas-1-thread)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.0171 (11.2) | 0.00614 (16.3) | numba-parallel | numba |
| 10 | 0.0296 (11.3) | 0.0129 (16.3) | numba-parallel | numba |
| 30 | 0.0919 (11.4) | 0.00745 (16.3) | numba-parallel | numba |
| 100 | 0.502 (11.8) | 0.0558 (16.4) | numba-parallel | numba |
| 300 | 4.42 (16) | 0.489 (16.8) | numba-parallel | numba |
| 1,000 | 88.5 (99.7) | 5.58 (22.1) | numba-parallel | numba-parallel |

