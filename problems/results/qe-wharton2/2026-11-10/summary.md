### P7 `kalman`, batch sweep, float64, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0158 (0.0163) | 0.00139 (2.47) | 0.00182 (3.39) | 0.00167 (0.137) | numba | numpy |
| 10 | 0.0289 (0.0297) | 0.0142 (2.49) | 0.00293 (3.38) | 0.00619 (0.129) | numba-parallel | numpy |
| 100 | 0.133 (0.133) | 0.14 (2.6) | 0.0164 (3.4) | 0.0775 (0.204) | numba-parallel | numpy |
| 1,000 | 1.15 (1.13) | 1.45 (3.92) | 0.14 (3.52) | 0.423 (0.55) | numba-parallel | jax-cpu |
| 10,000 | 10.8 (10.7) | 15.4 (17.6) | 1.57 (4.87) | 2.41 (2.52) | numba-parallel | jax-cpu |

### P7 `kalman`, batch sweep, float32, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0145 (0.0148) | 0.00177 (2.65) | 0.00208 (3.67) | 0.00497 (0.14) | numba | numpy |
| 10 | 0.0262 (0.0266) | 0.0175 (2.64) | 0.00313 (3.63) | 0.0163 (0.147) | numba-parallel | numpy |
| 100 | 0.132 (0.134) | 0.172 (2.77) | 0.017 (3.63) | 0.0971 (0.23) | numba-parallel | numpy |
| 1,000 | 1.18 (1.18) | 1.75 (4.3) | 0.155 (3.73) | 0.548 (0.674) | numba-parallel | jax-cpu |
| 10,000 | 11.6 (11.7) | 17.8 (20.6) | 1.53 (5.07) | 2.84 (2.95) | numba-parallel | jax-cpu |

### P7 `kalman`, batch100 sweep, float64, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0407 (0.0414) | 0.0284 (2.62) | 0.0293 (3.58) | 0.115 (0.26) | numba | numpy |
| 10 | 0.224 (0.228) | 0.285 (2.89) | 0.0944 (3.64) | 0.389 (0.531) | numba-parallel | numpy |
| 100 | 2.17 (2.17) | 2.87 (5.45) | 0.731 (4.3) | 1.8 (1.91) | numba-parallel | jax-cpu |
| 300 | 7 (7.06) | 8.73 (11.3) | 2.15 (5.66) | 5.32 (5.14) | numba-parallel | jax-cpu |
| 1,000 | 23.2 (23.3) |  | 7.72 (10.7) | 19 (19) | numba-parallel | numba-parallel |
| 3,000 | 68.9 (69) |  | 23.4 (25.3) | 57.1 (56.6) | numba-parallel | numba-parallel |
| 10,000 | 229 (229) |  | 78.1 (80.1) | 197 (195) | numba-parallel | numba-parallel |

### P7 `kalman`, batch100 sweep, float32, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0381 (0.04) | 0.0231 (2.71) | 0.0238 (3.73) | 0.0311 (0.171) | numba | numpy |
| 10 | 0.135 (0.137) | 0.232 (2.9) | 0.197 (3.9) | 0.227 (0.377) | numpy | numpy |
| 100 | 1.16 (1.17) | 2.32 (4.96) | 1.93 (5.58) | 1.12 (1.26) | jax-cpu | numpy |
| 300 | 3.66 (3.66) | 6.81 (9.49) | 5.78 (9.46) | 2.93 (3.05) | jax-cpu | jax-cpu |
| 1,000 | 12.5 (12.5) |  | 19.1 (22.7) | 10.9 (11.1) | jax-cpu | jax-cpu |
| 3,000 | 37 (37) |  | 56.8 (60.8) | 34.3 (34.4) | jax-cpu | jax-cpu |
| 10,000 | 122 (122) |  | 187 (191) | 118 (119) | jax-cpu | jax-cpu |

### P7 `kalman`, batch20 sweep, float64, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0173 (0.0177) | 0.00337 (2.63) | 0.00367 (3.47) | 0.00285 (0.14) | jax-cpu | numpy |
| 10 | 0.0385 (0.0399) | 0.0327 (2.56) | 0.0107 (3.49) | 0.032 (0.162) | numba-parallel | numpy |
| 100 | 0.216 (0.215) | 0.334 (2.87) | 0.0561 (3.53) | 0.175 (0.315) | numba-parallel | numpy |
| 300 | 0.553 (0.555) | 0.997 (3.52) | 0.16 (3.62) | 0.351 (0.485) | numba-parallel | jax-cpu |
| 1,000 | 1.88 (1.89) |  | 0.548 (4.15) | 0.733 (0.864) | numba-parallel | jax-cpu |
| 3,000 | 5.32 (5.33) |  | 0.928 (5.19) | 1.92 (2.03) | numba-parallel | jax-cpu |
| 10,000 | 17.6 (17.5) |  | 2.95 (8.86) | 6.37 (6.5) | numba-parallel | jax-cpu |

### P7 `kalman`, batch20 sweep, float32, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0183 (0.0181) | 0.0036 (2.62) | 0.00384 (3.61) | 0.00644 (0.138) | numba | numpy |
| 10 | 0.0418 (0.0426) | 0.035 (2.63) | 0.00534 (3.6) | 0.0345 (0.164) | numba-parallel | numpy |
| 100 | 0.207 (0.207) | 0.349 (2.94) | 0.0305 (3.62) | 0.202 (0.336) | numba-parallel | numpy |
| 300 | 0.552 (0.555) | 1.07 (3.65) | 0.0888 (3.66) | 0.383 (0.512) | numba-parallel | jax-cpu |
| 1,000 | 1.82 (1.81) |  | 0.279 (3.97) | 0.882 (1.02) | numba-parallel | jax-cpu |
| 3,000 | 5.32 (5.27) |  | 0.84 (4.47) | 1.81 (1.94) | numba-parallel | jax-cpu |
| 10,000 | 17.2 (17.5) |  | 2.83 (6.47) | 5.55 (5.35) | numba-parallel | jax-cpu |

### P7 `kalman`, batch50 sweep, float64, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0223 (0.0229) | 0.00881 (2.55) | 0.00905 (3.51) | 0.0167 (0.153) | numba | numpy |
| 10 | 0.0846 (0.0829) | 0.087 (2.62) | 0.0194 (3.49) | 0.144 (0.282) | numba-parallel | numpy |
| 100 | 0.57 (0.569) | 0.873 (3.4) | 0.144 (3.63) | 0.544 (0.68) | numba-parallel | numpy |
| 300 | 1.7 (1.73) | 2.64 (5.19) | 0.423 (3.92) | 1.12 (1.24) | numba-parallel | jax-cpu |
| 1,000 | 6.12 (6.13) |  | 1.36 (4.95) | 3.49 (3.51) | numba-parallel | jax-cpu |
| 3,000 | 18.4 (18.5) |  | 4.29 (7.76) | 11.3 (11.5) | numba-parallel | numba-parallel |
| 10,000 | 60.8 (61) |  | 14.8 (17.1) | 37.5 (37.3) | numba-parallel | numba-parallel |

### P7 `kalman`, batch50 sweep, float32, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.0242 (0.0247) | 0.00923 (2.68) | 0.00955 (3.71) | 0.0123 (0.145) | numba | numpy |
| 10 | 0.076 (0.0768) | 0.0917 (2.74) | 0.0136 (3.69) | 0.105 (0.24) | numba-parallel | numpy |
| 100 | 0.547 (0.544) | 0.93 (3.56) | 0.0882 (3.73) | 0.429 (0.572) | numba-parallel | numpy |
| 300 | 1.56 (1.57) | 2.78 (5.4) | 0.265 (3.89) | 0.918 (1.05) | numba-parallel | jax-cpu |
| 1,000 | 5.19 (5.25) |  | 0.846 (4.53) | 2.14 (2.28) | numba-parallel | jax-cpu |
| 3,000 | 15.7 (15.8) |  | 2.45 (6.21) | 7.28 (7.42) | numba-parallel | numba-parallel |
| 10,000 | 51.6 (52.2) |  | 8.37 (11.9) | 25.3 (25.5) | numba-parallel | numba-parallel |

### P7 `kalman`, state sweep, float64, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0122 (0.0124) | 0.00109 (2.6) | 0.00146 (3.56) | 0.000653 (0.126) | jax-cpu | numpy |
| 5 | 0.0123 (0.0127) | 0.00129 (2.58) | 0.00157 (3.54) | 0.00133 (0.128) | numba | numpy |
| 10 | 0.0158 (0.0163) | 0.00158 (2.57) | 0.00184 (3.53) | 0.00176 (0.143) | numba | numpy |
| 20 | 0.0174 (0.0178) | 0.00335 (2.57) | 0.0037 (3.52) | 0.00287 (0.139) | jax-cpu | numpy |
| 50 | 0.0224 (0.0231) | 0.00874 (2.56) | 0.00904 (3.52) | 0.0167 (0.156) | numba | numpy |
| 100 | 0.0411 (0.0423) | 0.0291 (2.58) | 0.0296 (3.53) | 0.115 (0.256) | numba | numpy |
| 200 | 0.128 (0.128) | 0.117 (2.66) | 0.117 (3.61) | 0.552 (0.679) | numba | numpy |

### P7 `kalman`, state sweep, float32, on qe-wharton2 (kalman-grid)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 2 | 0.0139 (0.0142) | 0.00109 (2.6) | 0.00139 (3.6) | 0.000655 (0.122) | jax-cpu | numpy |
| 5 | 0.0137 (0.014) | 0.00122 (2.61) | 0.0015 (3.6) | 0.00128 (0.124) | numba | numpy |
| 10 | 0.0142 (0.0146) | 0.00174 (2.59) | 0.00201 (3.6) | 0.00497 (0.138) | numba | numpy |
| 20 | 0.0182 (0.0185) | 0.00355 (2.6) | 0.00393 (3.6) | 0.00646 (0.136) | numba | numpy |
| 50 | 0.0236 (0.024) | 0.00928 (2.59) | 0.00966 (3.6) | 0.0123 (0.146) | numba | numpy |
| 100 | 0.0332 (0.0343) | 0.0232 (2.61) | 0.0235 (3.61) | 0.0316 (0.165) | numba | numpy |
| 200 | 0.0675 (0.0701) | 0.0747 (2.67) | 0.074 (3.67) | 0.144 (0.27) | numpy | numpy |

