### P7 `kalman`, batch sweep, float64, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.012 (0.0121) | 0.00138 (2.23) | 0.00155 (3.07) | 0.00161 (0.121) | numba | numpy |
| 10 | 0.0228 (0.0235) | 0.0137 (2.24) | 0.00297 (3.06) | 0.00598 (0.11) | numba-parallel | numpy |
| 100 | 0.122 (0.124) | 0.136 (2.35) | 0.0221 (3.08) | 0.0592 (0.171) | numba-parallel | numpy |
| 1,000 | 1.08 (1.09) | 1.38 (3.6) | 0.212 (3.29) | 0.389 (0.497) | numba-parallel | jax-cpu |
| 10,000 | 10.5 (10.5) | 13.9 (16.1) | 2.27 (5.26) | 2.57 (2.68) | numba-parallel | jax-cpu |

### P7 `kalman`, batch sweep, float32, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0123 (0.0126) | 0.00163 (2.3) | 0.00183 (3.21) | 0.00684 (0.118) | numba | numpy |
| 10 | 0.0236 (0.0246) | 0.0162 (2.32) | 0.00308 (3.2) | 0.0161 (0.129) | numba-parallel | numpy |
| 100 | 0.131 (0.132) | 0.161 (2.47) | 0.0249 (3.22) | 0.0832 (0.198) | numba-parallel | numpy |
| 1,000 | 1.18 (1.18) | 1.63 (3.93) | 0.237 (3.44) | 0.558 (0.66) | numba-parallel | jax-cpu |
| 10,000 | 11.7 (11.8) | 16.2 (18.4) | 2.35 (5.46) | 2.94 (3.08) | numba-parallel | jax-cpu |

### P7 `kalman`, batch100 sweep, float64, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.034 (0.0344) | 0.0264 (2.28) | 0.0265 (3.13) | 0.109 (0.226) | numba | numpy |
| 10 | 0.199 (0.199) | 0.26 (2.52) | 0.0966 (3.19) | 0.522 (0.637) | numba-parallel | numpy |
| 100 | 1.95 (1.95) | 2.59 (4.84) | 0.945 (4.01) | 2.32 (2.42) | numba-parallel | numpy |
| 300 | 6.7 (6.65) | 7.89 (10.2) | 2.86 (5.92) | 7.65 (7.77) | numba-parallel | numba-parallel |
| 1,000 | 22.3 (22.4) |  | 11.6 (14.3) | 32.4 (32.3) | numba-parallel | numba-parallel |
| 3,000 | 66.8 (68.3) |  | 34.8 (37.6) | 103 (102) | numba-parallel | numba-parallel |
| 10,000 | 224 (224) |  | 116 (119) | 378 (377) | numba-parallel | numba-parallel |

### P7 `kalman`, batch100 sweep, float32, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0274 (0.0287) | 0.019 (2.35) | 0.0194 (3.23) | 0.0264 (0.141) | numba | numpy |
| 10 | 0.118 (0.12) | 0.19 (2.49) | 0.127 (3.33) | 0.213 (0.333) | numpy | numpy |
| 100 | 1 (1.01) | 1.92 (4.23) | 1.27 (4.46) | 1.41 (1.52) | numpy | numpy |
| 300 | 3.47 (3.47) | 5.73 (8) | 3.85 (7.05) | 3.53 (3.67) | numpy | numpy |
| 1,000 | 12.1 (12.5) |  | 14.1 (17.3) | 13.9 (14.2) | numpy | numpy |
| 3,000 | 36 (36.7) |  | 42.1 (45.2) | 42.4 (42.3) | numpy | numpy |
| 10,000 | 119 (121) |  | 138 (142) | 147 (143) | numpy | numpy |

### P7 `kalman`, batch20 sweep, float64, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0132 (0.0136) | 0.00306 (2.26) | 0.00313 (3.1) | 0.0026 (0.115) | jax-cpu | numpy |
| 10 | 0.0296 (0.0303) | 0.0303 (2.27) | 0.00842 (3.1) | 0.0258 (0.139) | numba-parallel | numpy |
| 100 | 0.176 (0.178) | 0.301 (2.55) | 0.0721 (3.17) | 0.149 (0.267) | numba-parallel | numpy |
| 300 | 0.495 (0.496) | 0.913 (3.16) | 0.211 (3.3) | 0.343 (0.451) | numba-parallel | jax-cpu |
| 1,000 | 1.62 (1.66) |  | 0.686 (3.8) | 0.896 (1) | numba-parallel | jax-cpu |
| 3,000 | 4.85 (4.9) |  | 2.06 (5.19) | 2.12 (2.23) | numba-parallel | jax-cpu |
| 10,000 | 16.4 (16.4) |  | 7.07 (10.1) | 8.04 (8.26) | numba-parallel | jax-cpu |

### P7 `kalman`, batch20 sweep, float32, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.014 (0.0145) | 0.0039 (2.31) | 0.00338 (3.2) | 0.00755 (0.118) | numba-parallel | numpy |
| 10 | 0.0317 (0.0327) | 0.0349 (2.34) | 0.00562 (3.21) | 0.0273 (0.137) | numba-parallel | numpy |
| 100 | 0.188 (0.19) | 0.348 (2.65) | 0.0406 (3.25) | 0.161 (0.27) | numba-parallel | numpy |
| 300 | 0.546 (0.547) | 1.11 (3.42) | 0.12 (3.33) | 0.357 (0.468) | numba-parallel | jax-cpu |
| 1,000 | 1.76 (1.82) |  | 0.384 (3.59) | 0.886 (0.995) | numba-parallel | jax-cpu |
| 3,000 | 5.11 (5.11) |  | 1.18 (4.38) | 1.98 (2.21) | numba-parallel | jax-cpu |
| 10,000 | 17.1 (17.2) |  | 3.75 (7.11) | 6.18 (6.37) | numba-parallel | jax-cpu |

### P7 `kalman`, batch50 sweep, float64, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0183 (0.0185) | 0.00815 (2.27) | 0.00826 (3.1) | 0.0155 (0.129) | numba | numpy |
| 10 | 0.0644 (0.0655) | 0.0818 (2.33) | 0.0233 (3.11) | 0.123 (0.234) | numba-parallel | numpy |
| 100 | 0.504 (0.505) | 0.816 (3.06) | 0.21 (3.31) | 0.623 (0.724) | numba-parallel | numpy |
| 300 | 1.5 (1.52) | 2.43 (4.69) | 0.626 (3.73) | 1.31 (1.41) | numba-parallel | jax-cpu |
| 1,000 | 5.73 (5.71) |  | 2.07 (5.17) | 4.67 (4.8) | numba-parallel | jax-cpu |
| 3,000 | 17.3 (17.2) |  | 6.28 (9.25) | 15.8 (16.1) | numba-parallel | numba-parallel |
| 10,000 | 57 (56.8) |  | 20.9 (23.9) | 53.4 (53.7) | numba-parallel | numba-parallel |

### P7 `kalman`, batch50 sweep, float32, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0179 (0.0184) | 0.008 (2.31) | 0.00809 (3.21) | 0.0113 (0.122) | numba | numpy |
| 10 | 0.0629 (0.0634) | 0.0801 (2.38) | 0.0163 (3.22) | 0.0794 (0.193) | numba-parallel | numpy |
| 100 | 0.495 (0.496) | 0.798 (3.09) | 0.143 (3.34) | 0.436 (0.554) | numba-parallel | numpy |
| 300 | 1.43 (1.43) | 2.4 (4.69) | 0.406 (3.6) | 0.956 (1.07) | numba-parallel | jax-cpu |
| 1,000 | 4.9 (4.9) |  | 1.33 (4.51) | 2.57 (2.69) | numba-parallel | jax-cpu |
| 3,000 | 14.7 (14.7) |  | 3.97 (7.16) | 9.61 (9.73) | numba-parallel | numba-parallel |
| 10,000 | 49.2 (49.3) |  | 13.1 (16.4) | 35.4 (34.6) | numba-parallel | numba-parallel |

### P7 `kalman`, state sweep, float64, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0103 (0.0106) | 0.00098 (2.25) | 0.00114 (3.09) | 0.000589 (0.104) | jax-cpu | numpy |
| 5 | 0.0106 (0.0108) | 0.0011 (2.25) | 0.0013 (3.09) | 0.00114 (0.106) | numba | numpy |
| 10 | 0.0115 (0.0117) | 0.00144 (2.25) | 0.00155 (3.1) | 0.0016 (0.116) | numba | numpy |
| 20 | 0.0133 (0.0137) | 0.00303 (2.25) | 0.00319 (3.1) | 0.00266 (0.117) | jax-cpu | numpy |
| 50 | 0.0182 (0.0184) | 0.00807 (2.26) | 0.00821 (3.11) | 0.0155 (0.132) | numba | numpy |
| 100 | 0.0339 (0.0342) | 0.0264 (2.27) | 0.027 (3.12) | 0.11 (0.227) | numba | numpy |
| 200 | 0.111 (0.108) | 0.107 (2.36) | 0.105 (3.2) | 0.521 (0.624) | numba-parallel | numpy |

### P7 `kalman`, state sweep, float32, on qe-mini1 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0117 (0.012) | 0.00102 (2.3) | 0.00117 (3.21) | 0.000582 (0.101) | jax-cpu | numpy |
| 5 | 0.0117 (0.012) | 0.0011 (2.31) | 0.00135 (3.2) | 0.00113 (0.106) | numba | numpy |
| 10 | 0.0123 (0.0126) | 0.00161 (2.3) | 0.00177 (3.2) | 0.0071 (0.115) | numba | numpy |
| 20 | 0.014 (0.0143) | 0.00357 (2.3) | 0.00341 (3.2) | 0.00771 (0.118) | numba-parallel | numpy |
| 50 | 0.018 (0.0184) | 0.00811 (2.3) | 0.00824 (3.21) | 0.0116 (0.122) | numba | numpy |
| 100 | 0.0269 (0.0276) | 0.0192 (2.32) | 0.0192 (3.23) | 0.0263 (0.142) | numba-parallel | numpy |
| 200 | 0.0586 (0.0592) | 0.0655 (2.36) | 0.0665 (3.26) | 0.113 (0.219) | numpy | numpy |

