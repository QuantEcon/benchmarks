### P2 `hpi`, grid sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 0.0453 (0.046) | 0.0288 (1.37) | 0.0259 (1.5) | 0.0277 (0.101) | numba-parallel | numpy |
| 200 | 0.104 (0.11) | 0.0683 (1.38) | 0.0466 (1.52) | 0.0601 (0.135) | numba-parallel | numpy |
| 400 | 0.3 (0.322) | 0.214 (1.53) | 0.0967 (1.57) | 0.126 (0.219) | numba-parallel | jax-cpu |
| 700 | 0.763 (0.827) | 0.565 (1.89) | 0.182 (1.64) | 0.241 (0.323) | numba-parallel | jax-cpu |
| 1,000 | 1.44 (1.58) | 1.08 (2.43) | 0.29 (1.73) | 0.385 (0.485) | numba-parallel | jax-cpu |
| 1,500 | 3.37 (3.71) | 2.54 (3.92) | 0.536 (1.99) | 0.691 (0.787) | numba-parallel | jax-cpu |

### P2 `hpi`, income sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 25 | 0.022 (0.0234) | 0.0126 (1.32) | 0.0112 (1.48) | 0.00801 (0.0882) | jax-cpu | numpy |
| 50 | 0.0376 (0.04) | 0.0236 (1.34) | 0.0193 (1.48) | 0.0146 (0.0905) | jax-cpu | numpy |
| 100 | 0.0802 (0.0833) | 0.0463 (1.35) | 0.0362 (1.5) | 0.0419 (0.115) | numba-parallel | numpy |
| 200 | 0.141 (0.148) | 0.0901 (1.41) | 0.0688 (1.53) | 0.0948 (0.173) | numba-parallel | numpy |
| 400 | 0.303 (0.317) | 0.198 (1.51) | 0.156 (1.62) | 0.128 (0.208) | jax-cpu | jax-cpu |

### P5 `stationary`, dense sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 0.000169 (0.000272) | 0.000127 (1.23) | 0.000408 (1.37) | 0.000129 (0.0387) | numba | numpy |
| 200 | 0.000621 (0.00332) | 0.000533 (1.22) | 0.000799 (1.37) | 0.000452 (0.033) | jax-cpu | numpy |
| 500 | 0.006 (0.0103) | 0.00386 (1.22) | 0.0041 (1.38) | 0.00318 (0.0358) | jax-cpu | numpy |
| 1,000 | 0.0373 (0.0406) | 0.0356 (1.25) | 0.0357 (1.41) | 0.0229 (0.0575) | jax-cpu | numpy |
| 2,000 | 0.193 (0.207) | 0.273 (1.49) | 0.269 (1.66) | 0.155 (0.193) | jax-cpu | jax-cpu |
| 5,000 | 2.22 (2.34) | 3.23 (4.51) | 3.23 (4.65) | 2.02 (2.08) | jax-cpu | jax-cpu |

### P5 `stationary`, dense sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 0.000145 (0.000269) | 0.000115 (1.24) | 0.000388 (1.39) | 0.000132 (0.034) | numba | numpy |
| 200 | 0.000453 (0.000607) | 0.00039 (1.24) | 0.000682 (1.39) | 0.000295 (0.04) | jax-cpu | numpy |
| 500 | 0.0034 (0.00391) | 0.00267 (1.24) | 0.00299 (1.4) | 0.00189 (0.0344) | jax-cpu | numpy |
| 1,000 | 0.0157 (0.017) | 0.012 (1.25) | 0.0123 (1.4) | 0.00898 (0.0426) | jax-cpu | numpy |
| 2,000 | 0.0859 (0.0923) | 0.124 (1.36) | 0.124 (1.52) | 0.0588 (0.0972) | jax-cpu | numpy |
| 5,000 | 0.793 (0.819) | 1.43 (2.67) | 1.43 (2.82) | 0.655 (0.703) | jax-cpu | jax-cpu |

### P5 `stationary`, sparse sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00166 (0.00181) | 0.00107 (1.02) | 0.0368 (1.51) | 0.00135 (0.0484) | numba | numpy |
| 10,000 | 0.0124 (0.0126) | 0.0112 (1.03) | 0.0392 (1.54) | 0.0172 (0.063) | numba | numpy |
| 100,000 | 0.122 (0.124) | 0.118 (1.13) | 0.0543 (1.53) | 0.127 (0.173) | numba-parallel | numpy |
| 1,000,000 | 1.24 (1.25) | 1.23 (2.25) | 0.258 (1.76) | 1.26 (1.32) | numba-parallel | numpy |
| 10,000,000 | 12.9 (13) | 13 (14.1) | 2.67 (4.19) | 12 (12.2) | numba-parallel | numba-parallel |

### P5 `stationary`, sparse sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.000943 (0.00117) | 0.000558 (1.06) | 0.0217 (1.55) | 0.000629 (0.0487) | numba | numpy |
| 10,000 | 0.00744 (0.00773) | 0.00606 (1.07) | 0.0237 (1.57) | 0.008 (0.054) | numba | numpy |
| 100,000 | 0.073 (0.0748) | 0.0663 (1.13) | 0.0339 (1.56) | 0.0713 (0.118) | numba-parallel | numpy |
| 1,000,000 | 0.748 (0.762) | 0.691 (1.76) | 0.123 (1.66) | 0.695 (0.746) | numba-parallel | jax-cpu |
| 10,000,000 | 7.69 (7.82) | 7.27 (8.39) | 1.19 (2.76) | 6.64 (6.73) | numba-parallel | numba-parallel |

- NumPy's iterations in float32, against its float64 count: 157 against 266 at 1,000; 165 against 274 at 10,000; 174 against 283 at 100,000; 182 against 291 at 1,000,000; 192 against 299 at 10,000,000

### P6 `time_iter`, grid sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 120 | 0.0909 (0.093) | 0.0403 (0.808) | 0.00552 (0.927) | 0.0391 (0.148) | numba-parallel | numpy |
| 1,000 | 1.24 (1.25) | 0.456 (1.22) | 0.0415 (0.921) | 0.375 (0.49) | numba-parallel | jax-cpu |
| 5,000 | 7.28 (7.37) | 4.31 (5.07) | 0.334 (1.2) | 2.36 (2.48) | numba-parallel | numba-parallel |
| 20,000 | 34.1 (34.3) | 20 (20.7) | 1.47 (2.36) | 11.2 (11.3) | numba-parallel | numba-parallel |
| 50,000 | 96.4 (96.9) | 52.6 (53.5) | 3.95 (4.78) | 10.7 (10.9) | numba-parallel | numba-parallel |

### P6 `time_iter`, grid sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 120 | 0.1 (0.103) | 0.0379 (0.817) | 0.00528 (0.913) | 0.0265 (0.139) | numba-parallel | numpy |
| 1,000 | 1.38 (1.39) | 0.431 (1.21) | 0.0395 (0.932) | 0.253 (0.371) | numba-parallel | jax-cpu |
| 5,000 | 8.69 (9.05) | 4.08 (4.83) | 0.309 (1.19) | 1.6 (1.72) | numba-parallel | numba-parallel |
| 20,000 | 48 (48) | 18.8 (19.5) | 1.36 (2.22) | 7.53 (7.67) | numba-parallel | numba-parallel |
| 50,000 | 163 (165) | 49.2 (49.9) | 3.54 (4.38) | 20 (20.2) | numba-parallel | numba-parallel |

### P6 `time_iter`, shocks sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 250 | 0.0908 (0.0929) | 0.0401 (0.815) | 0.00551 (0.902) | 0.0391 (0.149) | numba-parallel | numpy |
| 1,000 | 0.216 (0.217) | 0.159 (0.931) | 0.0155 (0.907) | 0.134 (0.243) | numba-parallel | numpy |
| 5,000 | 0.953 (0.972) | 0.78 (1.55) | 0.0667 (0.952) | 0.64 (0.75) | numba-parallel | jax-cpu |
| 20,000 | 4.03 (4.12) | 3.09 (3.84) | 0.265 (1.14) | 2.61 (2.72) | numba-parallel | numba-parallel |
| 100,000 | 20.3 (20.8) | 15.8 (16.6) | 1.31 (2.15) | 5 (5.25) | numba-parallel | numba-parallel |

### P6 `time_iter`, shocks sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 250 | 0.101 (0.104) | 0.0379 (0.821) | 0.00525 (0.904) | 0.0266 (0.141) | numba-parallel | numpy |
| 1,000 | 0.228 (0.23) | 0.152 (0.926) | 0.0149 (0.921) | 0.0931 (0.204) | numba-parallel | jax-cpu |
| 5,000 | 1 (1) | 0.745 (1.52) | 0.0658 (0.964) | 0.447 (0.556) | numba-parallel | jax-cpu |
| 20,000 | 4.11 (4.16) | 2.97 (3.74) | 0.25 (1.13) | 1.74 (1.86) | numba-parallel | numba-parallel |
| 100,000 | 20.4 (21) | 14.7 (15.5) | 1.16 (2) | 8.93 (9.06) | numba-parallel | numba-parallel |

### P8 `mle`, obs sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 10,000 | 0.00104 (0.00107) | 0.00077 (1.59) | 0.00192 (1.74) | 0.00101 (0.107) | numba | numpy |
| 100,000 | 0.0106 (0.0107) | 0.00889 (1.59) | 0.00769 (1.7) | 0.00912 (0.118) | numba-parallel | numpy |
| 500,000 | 0.103 (0.103) | 0.0905 (1.66) | 0.0733 (1.75) | 0.0606 (0.17) | jax-cpu | numpy |
| 1,000,000 | 0.145 (0.146) | 0.128 (1.7) | 0.101 (1.77) | 0.088 (0.198) | jax-cpu | numpy |
| 3,000,000 | 0.611 (0.617) | 0.545 (2.11) | 0.424 (2.11) | 0.369 (0.503) | jax-cpu | jax-cpu |
| 10,000,000 | 1.92 (1.92) | 1.69 (3.28) | 1.32 (2.98) | 1.15 (1.3) | jax-cpu | jax-cpu |

### P8 `mle`, obs sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 10,000 | 0.000933 (0.00104) | 0.000695 (1.58) | 0.00193 (1.69) | 0.0008 (0.11) | numba | numpy |
| 100,000 | 0.00944 (0.00968) | 0.00788 (1.58) | 0.00757 (1.69) | 0.00882 (0.117) | numba-parallel | numpy |
| 500,000 | 0.0648 (0.0656) | 0.0554 (1.63) | 0.0444 (1.73) | 0.0547 (0.163) | numba-parallel | numpy |
| 1,000,000 | 0.116 (0.117) | 0.1 (1.68) | 0.0832 (1.76) | 0.0757 (0.185) | jax-cpu | numpy |
| 3,000,000 | 0.515 (0.518) | 0.453 (2.02) | 0.386 (2.04) | 0.344 (0.457) | jax-cpu | jax-cpu |
| 10,000,000 | 1.61 (1.62) | 1.41 (2.97) | 1.15 (2.82) | 1.22 (1.34) | numba-parallel | jax-cpu |

- NumPy's iterations in float32, against its float64 count: 18 against 17 at 3,000,000; 17 against 16 at 10,000,000
- 10,000,000, `jax-cpu`: 19 iterations, against NumPy's 17 in float32

### P8 `mle`, params sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 3 | 0.00601 (0.00614) | 0.00504 (1.58) | 0.00446 (1.66) | 0.00535 (0.113) | numba-parallel | numpy |
| 10 | 0.00698 (0.00697) | 0.00528 (1.56) | 0.00399 (1.66) | 0.00332 (0.131) | jax-cpu | numpy |
| 30 | 0.0163 (0.0165) | 0.0159 (1.56) | 0.0124 (1.66) | 0.00917 (0.14) | jax-cpu | numpy |
| 100 | 0.084 (0.0844) | 0.0914 (1.65) | 0.0721 (1.74) | 0.0513 (0.187) | jax-cpu | numpy |
| 300 | 0.477 (0.474) | 0.526 (2.08) | 0.442 (2.09) | 0.564 (0.695) | numba-parallel | numpy |
| 1,000 | 7.29 (7.28) | 7.72 (9.31) | 7.1 (8.76) | 5.18 (4.64) | jax-cpu | jax-cpu |

### P8 `mle`, params sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 3 | 0.00555 (0.00572) | 0.00463 (1.61) | 0.00449 (1.72) | 0.00516 (0.115) | numba-parallel | numpy |
| 10 | 0.0074 (0.00773) | 0.00671 (1.61) | 0.00628 (1.73) | 0.00423 (0.133) | jax-cpu | numpy |
| 30 | 0.0103 (0.0107) | 0.0105 (1.62) | 0.00866 (1.72) | 0.00613 (0.132) | jax-cpu | numpy |
| 100 | 0.0383 (0.039) | 0.0447 (1.65) | 0.0318 (1.74) | 0.0252 (0.153) | jax-cpu | numpy |
| 300 | 0.167 (0.169) | 0.224 (1.83) | 0.15 (1.86) | 0.325 (0.456) | numba-parallel | numpy |
| 1,000 | 2.11 (2.11) | 2.61 (4.21) | 2.03 (3.74) | 2.54 (2.43) | numba-parallel | numpy |

### P8 `mle_hand`, obs sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 10,000 | 0.000866 (0.0989) | jax-cpu | jax-cpu |
| 100,000 | 0.00839 (0.108) | jax-cpu | jax-cpu |
| 500,000 | 0.0561 (0.155) | jax-cpu | jax-cpu |
| 1,000,000 | 0.0807 (0.179) | jax-cpu | jax-cpu |
| 3,000,000 | 0.345 (0.447) | jax-cpu | jax-cpu |
| 10,000,000 | 1.07 (1.19) | jax-cpu | jax-cpu |

### P8 `mle_hand`, obs sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 10,000 | 0.000749 (0.1) | jax-cpu | jax-cpu |
| 100,000 | 0.00821 (0.106) | jax-cpu | jax-cpu |
| 500,000 | 0.0524 (0.154) | jax-cpu | jax-cpu |
| 1,000,000 | 0.0726 (0.173) | jax-cpu | jax-cpu |
| 3,000,000 | 0.332 (0.433) | jax-cpu | jax-cpu |
| 10,000,000 | 1.15 (1.26) | jax-cpu | jax-cpu |

- NumPy's iterations in float32, against its float64 count: 18 against 17 at 3,000,000; 17 against 16 at 10,000,000
- 10,000,000, `jax-cpu`: 19 iterations, against NumPy's 17 in float32

### P8 `mle_hand`, params sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 3 | 0.00488 (0.103) | jax-cpu | jax-cpu |
| 10 | 0.00228 (0.126) | jax-cpu | jax-cpu |
| 30 | 0.00772 (0.127) | jax-cpu | jax-cpu |
| 100 | 0.0491 (0.174) | jax-cpu | jax-cpu |
| 300 | 0.572 (0.712) | jax-cpu | jax-cpu |
| 1,000 | 5.12 (4.68) | jax-cpu | jax-cpu |

### P8 `mle_hand`, params sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 3 | 0.00481 (0.106) | jax-cpu | jax-cpu |
| 10 | 0.00388 (0.122) | jax-cpu | jax-cpu |
| 30 | 0.00539 (0.122) | jax-cpu | jax-cpu |
| 100 | 0.024 (0.144) | jax-cpu | jax-cpu |
| 300 | 0.366 (0.479) | jax-cpu | jax-cpu |
| 1,000 | 2.47 (2.5) | jax-cpu | jax-cpu |

### P8 `mle_onepass`, obs sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.000649 (1.98) | 0.00146 (2.94) | numba | numba |
| 100,000 | 0.00791 (2) | 0.00239 (2.94) | numba-parallel | numba |
| 500,000 | 0.0547 (2.02) | 0.00727 (2.94) | numba-parallel | numba |
| 1,000,000 | 0.0783 (2.05) | 0.00909 (2.93) | numba-parallel | numba |
| 3,000,000 | 0.335 (2.29) | 0.0301 (2.97) | numba-parallel | numba |
| 10,000,000 | 1.04 (2.99) | 0.0889 (2.99) | numba-parallel | numba |

### P8 `mle_onepass`, obs sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.000581 (1.98) | 0.00155 (2.97) | numba | numba |
| 100,000 | 0.00685 (1.97) | 0.00223 (2.95) | numba-parallel | numba |
| 500,000 | 0.0482 (1.99) | 0.00676 (2.94) | numba-parallel | numba |
| 1,000,000 | 0.069 (2.01) | 0.0071 (2.94) | numba-parallel | numba |
| 3,000,000 | 0.29 (2.26) | 0.0251 (2.94) | numba-parallel | numba |
| 10,000,000 | 0.905 (2.83) | 0.0762 (2.98) | numba-parallel | numba |

- NumPy's iterations in float32, against its float64 count: 18 against 17 at 3,000,000; 17 against 16 at 10,000,000
- 3,000,000, `numba`: 17 iterations, against NumPy's 18 in float32
- 3,000,000, `numba-parallel`: 17 iterations, against NumPy's 18 in float32
- 10,000,000, `numba`: 16 iterations, against NumPy's 17 in float32
- 10,000,000, `numba-parallel`: 16 iterations, against NumPy's 17 in float32

### P8 `mle_onepass`, params sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.00448 (1.92) | 0.00137 (2.87) | numba-parallel | numba |
| 10 | 0.0154 (1.93) | 0.0024 (2.87) | numba-parallel | numba |
| 30 | 0.0612 (1.96) | 0.00695 (2.86) | numba-parallel | numba |
| 100 | 0.458 (2.36) | 0.0505 (2.91) | numba-parallel | numba |
| 300 | 5.71 (7.59) | 0.625 (3.46) | numba-parallel | numba-parallel |

### P8 `mle_onepass`, params sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.00403 (1.98) | 0.00134 (2.95) | numba-parallel | numba |
| 10 | 0.0115 (1.98) | 0.00204 (2.94) | numba-parallel | numba |
| 30 | 0.0429 (2) | 0.00664 (2.95) | numba-parallel | numba |
| 100 | 0.304 (2.27) | 0.0305 (2.97) | numba-parallel | numba |
| 300 | 3 (4.94) | 0.329 (3.25) | numba-parallel | numba-parallel |

### P8 `mle_blocked`, obs sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.000988 (2.47) | 0.00156 (3.59) | numba | numba |
| 100,000 | 0.0118 (2.46) | 0.00288 (3.6) | numba-parallel | numba |
| 500,000 | 0.0814 (2.51) | 0.00992 (3.58) | numba-parallel | numba |
| 1,000,000 | 0.115 (2.55) | 0.0112 (3.6) | numba-parallel | numba |
| 3,000,000 | 0.488 (2.91) | 0.0457 (3.61) | numba-parallel | numba |
| 10,000,000 | 1.54 (3.95) | 0.131 (3.67) | numba-parallel | numba-parallel |

### P8 `mle_blocked`, obs sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.000845 (2.47) | 0.0016 (3.63) | numba | numba |
| 100,000 | 0.00979 (2.46) | 0.0034 (3.63) | numba-parallel | numba |
| 500,000 | 0.0692 (2.52) | 0.0156 (3.64) | numba-parallel | numba |
| 1,000,000 | 0.0979 (2.55) | 0.0202 (3.63) | numba-parallel | numba |
| 3,000,000 | 0.415 (2.86) | 0.0817 (3.69) | numba-parallel | numba |
| 10,000,000 | 1.3 (3.74) | 0.255 (3.86) | numba-parallel | numba |

- NumPy's iterations in float32, against its float64 count: 18 against 17 at 3,000,000; 17 against 16 at 10,000,000
- 3,000,000, `numba`: 17 iterations, against NumPy's 18 in float32
- 3,000,000, `numba-parallel`: 17 iterations, against NumPy's 18 in float32
- 10,000,000, `numba`: 16 iterations, against NumPy's 17 in float32
- 10,000,000, `numba-parallel`: 16 iterations, against NumPy's 17 in float32

### P8 `mle_blocked`, params sweep, float64, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.00682 (2.43) | 0.00165 (3.56) | numba-parallel | numba |
| 10 | 0.0132 (2.43) | 0.00281 (3.59) | numba-parallel | numba |
| 30 | 0.0195 (2.43) | 0.00558 (3.56) | numba-parallel | numba |
| 100 | 0.0878 (2.5) | 0.0316 (3.58) | numba-parallel | numba |
| 300 | 0.516 (2.92) | 0.251 (3.78) | numba-parallel | numba |
| 1,000 | 8.11 (10.5) | 6.06 (9.49) | numba-parallel | numba-parallel |

### P8 `mle_blocked`, params sweep, float32, on qe-wharton2

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.00582 (2.52) | 0.002 (3.71) | numba-parallel | numba |
| 10 | 0.00926 (2.52) | 0.00213 (3.71) | numba-parallel | numba |
| 30 | 0.0131 (2.53) | 0.00261 (3.7) | numba-parallel | numba |
| 100 | 0.0315 (2.54) | 0.0118 (3.71) | numba-parallel | numba |
| 300 | 0.154 (2.67) | 0.0828 (3.76) | numba-parallel | numba |
| 1,000 | 2.24 (4.74) | 1.93 (5.62) | numba-parallel | numba |

