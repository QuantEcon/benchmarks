### P2 `hpi`, grid sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 0.0443 (0.0459) | 0.0243 (1.14) | 0.0216 (1.26) | 0.0267 (0.0974) | numba-parallel | numpy |
| 200 | 0.0765 (0.0809) | 0.0614 (1.17) | 0.0425 (1.28) | 0.0548 (0.126) | numba-parallel | numpy |
| 400 | 0.244 (0.258) | 0.199 (1.31) | 0.0999 (1.33) | 0.134 (0.218) | numba-parallel | jax-cpu |
| 700 | 0.662 (0.709) | 0.528 (1.64) | 0.201 (1.44) | 0.288 (0.365) | numba-parallel | jax-cpu |
| 1,000 | 1.25 (1.36) | 1.02 (2.15) | 0.34 (1.58) | 0.492 (0.575) | numba-parallel | jax-cpu |
| 1,500 | 3.01 (3.28) | 2.4 (3.56) | 0.692 (1.9) | 0.972 (1.08) | numba-parallel | jax-cpu |

### P2 `hpi`, income sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 25 | 0.0167 (0.0176) | 0.0111 (1.12) | 0.00948 (1.25) | 0.00799 (0.0795) | jax-cpu | numpy |
| 50 | 0.0271 (0.029) | 0.0197 (1.12) | 0.0157 (1.25) | 0.0145 (0.0814) | jax-cpu | numpy |
| 100 | 0.052 (0.0561) | 0.0408 (1.15) | 0.0319 (1.26) | 0.0397 (0.11) | numba-parallel | numpy |
| 200 | 0.104 (0.11) | 0.0839 (1.19) | 0.0642 (1.3) | 0.089 (0.164) | numba-parallel | numpy |
| 400 | 0.305 (0.314) | 0.185 (1.29) | 0.146 (1.38) | 0.149 (0.221) | numba-parallel | jax-cpu |

### P5 `stationary`, dense sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 0.000153 (0.00065) | 0.000122 (1.13) | 0.000226 (1.18) | 0.000109 (0.0336) | jax-cpu | numpy |
| 200 | 0.000631 (0.000749) | 0.000558 (1.03) | 0.000717 (1.16) | 0.000434 (0.0277) | jax-cpu | numpy |
| 500 | 0.00647 (0.00772) | 0.0039 (1.07) | 0.00403 (1.17) | 0.00267 (0.0307) | jax-cpu | numpy |
| 1,000 | 0.0365 (0.0399) | 0.0249 (1.07) | 0.0236 (1.19) | 0.0187 (0.0474) | jax-cpu | numpy |
| 2,000 | 0.18 (0.19) | 0.153 (1.21) | 0.152 (1.33) | 0.132 (0.163) | jax-cpu | jax-cpu |
| 5,000 | 2.01 (2.09) | 1.94 (3.04) | 1.93 (3.23) | 1.82 (1.87) | jax-cpu | jax-cpu |

### P5 `stationary`, dense sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 100 | 0.000143 (0.000228) | 0.000104 (1.1) | 0.000222 (1.21) | 0.000116 (0.0296) | numba | numpy |
| 200 | 0.000408 (0.000529) | 0.000378 (1.08) | 0.000482 (1.23) | 0.000262 (0.0342) | jax-cpu | numpy |
| 500 | 0.00284 (0.00321) | 0.0026 (1.09) | 0.00271 (1.22) | 0.00141 (0.0289) | jax-cpu | numpy |
| 1,000 | 0.0143 (0.0158) | 0.0114 (1.09) | 0.0115 (1.22) | 0.00602 (0.035) | jax-cpu | numpy |
| 2,000 | 0.0754 (0.0788) | 0.0791 (1.16) | 0.0768 (1.28) | 0.0431 (0.074) | jax-cpu | jax-cpu |
| 5,000 | 0.706 (0.719) | 0.653 (1.73) | 0.656 (1.86) | 0.573 (0.605) | jax-cpu | jax-cpu |

### P5 `stationary`, sparse sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.00142 (0.00431) | 0.000778 (0.87) | 0.0147 (1.29) | 0.00126 (0.0461) | numba | numpy |
| 10,000 | 0.0112 (0.0115) | 0.00865 (0.886) | 0.0206 (1.3) | 0.0146 (0.0573) | numba | numpy |
| 100,000 | 0.112 (0.113) | 0.093 (0.963) | 0.042 (1.31) | 0.119 (0.16) | numba-parallel | numpy |
| 1,000,000 | 1.13 (1.14) | 0.953 (1.84) | 0.344 (1.65) | 1.35 (1.39) | numba-parallel | numpy |
| 10,000,000 | 12 (12) | 10.5 (11.3) | 4.33 (5.6) | 14.4 (14.2) | numba-parallel | numba-parallel |

### P5 `stationary`, sparse sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 1,000 | 0.000956 (0.001) | 0.000534 (0.892) | 0.00882 (1.3) | 0.000625 (0.0446) | numba | numpy |
| 10,000 | 0.00692 (0.007) | 0.00598 (0.889) | 0.0125 (1.3) | 0.00715 (0.0483) | numba | numpy |
| 100,000 | 0.0712 (0.0745) | 0.0702 (0.945) | 0.0282 (1.31) | 0.066 (0.106) | numba-parallel | numpy |
| 1,000,000 | 0.723 (0.736) | 0.676 (1.56) | 0.161 (1.43) | 0.683 (0.723) | numba-parallel | jax-cpu |
| 10,000,000 | 7.5 (7.53) | 6.94 (7.82) | 1.96 (3.24) | 7.34 (7.32) | numba-parallel | numba-parallel |

- NumPy's iterations in float32, against its float64 count: 157 against 266 at 1,000; 165 against 274 at 10,000; 174 against 283 at 100,000; 182 against 291 at 1,000,000; 192 against 299 at 10,000,000

### P6 `time_iter`, grid sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 120 | 0.0802 (0.0857) | 0.0369 (0.684) | 0.00707 (0.752) | 0.0352 (0.135) | numba-parallel | numpy |
| 1,000 | 1.13 (1.13) | 0.427 (1.07) | 0.0725 (0.809) | 0.36 (0.466) | numba-parallel | jax-cpu |
| 5,000 | 6.99 (7.09) | 3.96 (4.59) | 0.615 (1.29) | 2.33 (2.68) | numba-parallel | numba-parallel |
| 20,000 | 32.4 (32.8) | 18.9 (19.7) | 3.03 (3.49) | 11 (11.7) | numba-parallel | numba-parallel |
| 50,000 | 90 (91.5) | 51.2 (52) | 8.09 (8.34) | 19.5 (19.6) | numba-parallel | numba-parallel |

### P6 `time_iter`, grid sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 120 | 0.0991 (0.103) | 0.0362 (0.723) | 0.00704 (0.771) | 0.0241 (0.135) | numba-parallel | numpy |
| 1,000 | 1.25 (1.25) | 0.401 (1.06) | 0.0691 (0.815) | 0.248 (0.378) | numba-parallel | jax-cpu |
| 5,000 | 8.18 (8.2) | 3.84 (4.48) | 0.555 (1.26) | 1.48 (1.77) | numba-parallel | numba-parallel |
| 20,000 | 46.8 (46.3) | 18.4 (19.1) | 2.94 (3.33) | 7.14 (8.28) | numba-parallel | numba-parallel |
| 50,000 | 159 (157) | 48 (49) | 7.39 (7.63) | 19.4 (20.3) | numba-parallel | numba-parallel |

### P6 `time_iter`, shocks sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 250 | 0.08 (0.0804) | 0.0376 (0.69) | 0.00705 (0.755) | 0.0354 (0.138) | numba-parallel | numpy |
| 1,000 | 0.193 (0.199) | 0.149 (0.795) | 0.0258 (0.771) | 0.133 (0.234) | numba-parallel | numpy |
| 5,000 | 0.873 (0.888) | 0.716 (1.36) | 0.127 (0.861) | 0.641 (0.81) | numba-parallel | jax-cpu |
| 20,000 | 3.81 (3.84) | 2.84 (3.48) | 0.523 (1.2) | 2.62 (3.22) | numba-parallel | numba-parallel |
| 100,000 | 19.9 (20) | 14.8 (15.5) | 3.09 (3.29) | 9.19 (9.39) | numba-parallel | numba-parallel |

### P6 `time_iter`, shocks sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 250 | 0.0995 (0.104) | 0.0358 (0.721) | 0.00676 (0.762) | 0.0239 (0.13) | numba-parallel | numpy |
| 1,000 | 0.207 (0.208) | 0.142 (0.797) | 0.024 (0.777) | 0.0888 (0.19) | numba-parallel | jax-cpu |
| 5,000 | 0.917 (0.934) | 0.694 (1.34) | 0.116 (0.858) | 0.443 (0.572) | numba-parallel | jax-cpu |
| 20,000 | 3.85 (3.88) | 2.75 (3.4) | 0.518 (1.18) | 1.71 (2.08) | numba-parallel | numba-parallel |
| 100,000 | 19.7 (19.7) | 13.8 (14.4) | 2.72 (2.94) | 9.05 (10.2) | numba-parallel | numba-parallel |

### P8 `mle`, obs sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 10,000 | 0.000931 (0.00101) | 0.000734 (1.33) | 0.00123 (1.42) | 0.000903 (0.0836) | numba | numpy |
| 100,000 | 0.0104 (0.0106) | 0.00959 (1.33) | 0.00828 (1.42) | 0.00866 (0.0911) | numba-parallel | numpy |
| 500,000 | 0.1 (0.102) | 0.0858 (1.4) | 0.0701 (1.47) | 0.0577 (0.14) | jax-cpu | numpy |
| 1,000,000 | 0.137 (0.139) | 0.123 (1.45) | 0.0969 (1.5) | 0.086 (0.171) | jax-cpu | numpy |
| 3,000,000 | 0.582 (0.584) | 0.523 (1.84) | 0.41 (1.81) | 0.386 (0.473) | jax-cpu | jax-cpu |
| 10,000,000 | 1.83 (1.84) | 1.65 (2.97) | 1.3 (2.7) | 1.24 (1.33) | jax-cpu | jax-cpu |

### P8 `mle`, obs sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 10,000 | 0.000826 (0.00091) | 0.000678 (1.33) | 0.00121 (1.42) | 0.000728 (0.0871) | numba | numpy |
| 100,000 | 0.00915 (0.00931) | 0.00787 (1.34) | 0.00719 (1.43) | 0.0083 (0.0922) | numba-parallel | numpy |
| 500,000 | 0.0631 (0.0644) | 0.0539 (1.39) | 0.0434 (1.47) | 0.0537 (0.139) | numba-parallel | numpy |
| 1,000,000 | 0.111 (0.112) | 0.0982 (1.44) | 0.0808 (1.5) | 0.0738 (0.16) | jax-cpu | numpy |
| 3,000,000 | 0.485 (0.488) | 0.445 (1.78) | 0.36 (1.78) | 0.34 (0.425) | jax-cpu | jax-cpu |
| 10,000,000 | 1.51 (1.51) | 1.41 (2.73) | 1.13 (2.55) | 1.21 (1.31) | numba-parallel | jax-cpu |

- NumPy's iterations in float32, against its float64 count: 18 against 17 at 3,000,000; 17 against 16 at 10,000,000
- 10,000,000, `jax-cpu`: 19 iterations, against NumPy's 17 in float32

### P8 `mle`, params sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 3 | 0.00617 (0.00626) | 0.00557 (1.32) | 0.00483 (1.41) | 0.00511 (0.087) | numba-parallel | numpy |
| 10 | 0.00622 (0.00651) | 0.0049 (1.35) | 0.00345 (1.41) | 0.00356 (0.108) | numba-parallel | numpy |
| 30 | 0.0163 (0.0167) | 0.0153 (1.33) | 0.0121 (1.41) | 0.012 (0.116) | jax-cpu | numpy |
| 100 | 0.0747 (0.0755) | 0.0818 (1.4) | 0.0646 (1.46) | 0.0764 (0.178) | numba-parallel | numpy |
| 300 | 0.404 (0.42) | 0.46 (1.77) | 0.379 (1.77) | 0.734 (0.694) | numba-parallel | numpy |
| 1,000 | 5.87 (6.1) | 6.35 (7.82) | 5.79 (7.15) | 11.3 (11.6) | numba-parallel | numpy |

### P8 `mle`, params sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 3 | 0.00926 (0.00946) | 0.00517 (1.63) | 0.00418 (1.47) | 0.00492 (0.0889) | numba-parallel | numpy |
| 10 | 0.00685 (0.00665) | 0.00641 (1.34) | 0.00583 (1.42) | 0.00406 (0.107) | jax-cpu | numpy |
| 30 | 0.00945 (0.00973) | 0.0098 (1.34) | 0.00782 (1.43) | 0.00636 (0.104) | jax-cpu | numpy |
| 100 | 0.0355 (0.0358) | 0.0419 (1.37) | 0.03 (1.45) | 0.0358 (0.137) | numba-parallel | numpy |
| 300 | 0.146 (0.147) | 0.201 (1.53) | 0.135 (1.55) | 0.357 (0.466) | numba-parallel | numpy |
| 1,000 | 1.77 (1.88) | 2.25 (3.59) | 1.74 (3.16) | 5.07 (4.65) | numba-parallel | numpy |

### P8 `mle_hand`, obs sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 10,000 | 0.0011 (0.108) | jax-cpu | jax-cpu |
| 100,000 | 0.0101 (0.104) | jax-cpu | jax-cpu |
| 500,000 | 0.0621 (0.155) | jax-cpu | jax-cpu |
| 1,000,000 | 0.087 (0.166) | jax-cpu | jax-cpu |
| 3,000,000 | 0.338 (0.448) | jax-cpu | jax-cpu |
| 10,000,000 | 1.07 (1.14) | jax-cpu | jax-cpu |

### P8 `mle_hand`, obs sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 10,000 | 0.000684 (0.0757) | jax-cpu | jax-cpu |
| 100,000 | 0.0077 (0.0835) | jax-cpu | jax-cpu |
| 500,000 | 0.0501 (0.126) | jax-cpu | jax-cpu |
| 1,000,000 | 0.0692 (0.145) | jax-cpu | jax-cpu |
| 3,000,000 | 0.318 (0.393) | jax-cpu | jax-cpu |
| 10,000,000 | 1.12 (1.2) | jax-cpu | jax-cpu |

- NumPy's iterations in float32, against its float64 count: 18 against 17 at 3,000,000; 17 against 16 at 10,000,000
- 10,000,000, `jax-cpu`: 19 iterations, against NumPy's 17 in float32

### P8 `mle_hand`, params sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 3 | 0.0046 (0.0786) | jax-cpu | jax-cpu |
| 10 | 0.00234 (0.0989) | jax-cpu | jax-cpu |
| 30 | 0.0104 (0.105) | jax-cpu | jax-cpu |
| 100 | 0.0734 (0.173) | jax-cpu | jax-cpu |
| 300 | 0.637 (0.651) | jax-cpu | jax-cpu |
| 1,000 | 10.2 (9.89) | jax-cpu | jax-cpu |

### P8 `mle_hand`, params sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 3 | 0.00632 (0.118) | jax-cpu | jax-cpu |
| 10 | 0.00472 (0.12) | jax-cpu | jax-cpu |
| 30 | 0.00636 (0.115) | jax-cpu | jax-cpu |
| 100 | 0.0354 (0.144) | jax-cpu | jax-cpu |
| 300 | 0.305 (0.411) | jax-cpu | jax-cpu |
| 1,000 | 4.88 (4.6) | jax-cpu | jax-cpu |

### P8 `mle_onepass`, obs sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.000711 (2.02) | 0.000823 (2.5) | numba | numba |
| 100,000 | 0.0092 (1.66) | 0.00201 (2.49) | numba-parallel | numba |
| 500,000 | 0.061 (1.72) | 0.0114 (2.48) | numba-parallel | numba |
| 1,000,000 | 0.0818 (1.73) | 0.0154 (2.49) | numba-parallel | numba |
| 3,000,000 | 0.344 (1.99) | 0.066 (2.53) | numba-parallel | numba |
| 10,000,000 | 1.08 (2.87) | 0.21 (2.66) | numba-parallel | numba-parallel |

### P8 `mle_onepass`, obs sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.000711 (1.92) | 0.000788 (2.51) | numba | numba |
| 100,000 | 0.00689 (1.67) | 0.00177 (2.51) | numba-parallel | numba |
| 500,000 | 0.0497 (1.71) | 0.00996 (2.51) | numba-parallel | numba |
| 1,000,000 | 0.0688 (1.73) | 0.0124 (2.52) | numba-parallel | numba |
| 3,000,000 | 0.294 (1.96) | 0.0528 (2.55) | numba-parallel | numba |
| 10,000,000 | 0.922 (2.66) | 0.167 (2.66) | numba-parallel | numba-parallel |

- NumPy's iterations in float32, against its float64 count: 18 against 17 at 3,000,000; 17 against 16 at 10,000,000
- 3,000,000, `numba`: 17 iterations, against NumPy's 18 in float32
- 3,000,000, `numba-parallel`: 17 iterations, against NumPy's 18 in float32
- 10,000,000, `numba`: 16 iterations, against NumPy's 17 in float32
- 10,000,000, `numba-parallel`: 16 iterations, against NumPy's 17 in float32

### P8 `mle_onepass`, params sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.00466 (1.89) | 0.00118 (2.47) | numba-parallel | numba |
| 10 | 0.0153 (1.66) | 0.00266 (2.47) | numba-parallel | numba |
| 30 | 0.0607 (1.7) | 0.00978 (2.47) | numba-parallel | numba |
| 100 | 0.462 (2.1) | 0.0905 (2.56) | numba-parallel | numba |
| 300 | 5.45 (7.38) | 1.67 (3.76) | numba-parallel | numba-parallel |

### P8 `mle_onepass`, params sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.00396 (2.01) | 0.00105 (2.53) | numba-parallel | numba |
| 10 | 0.0113 (1.68) | 0.00207 (2.51) | numba-parallel | numba |
| 30 | 0.0427 (1.71) | 0.0072 (2.5) | numba-parallel | numba |
| 100 | 0.287 (1.95) | 0.0506 (2.55) | numba-parallel | numba |
| 300 | 2.94 (4.69) | 0.831 (3.08) | numba-parallel | numba-parallel |

### P8 `mle_blocked`, obs sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.00107 (2.43) | 0.000854 (3.05) | numba-parallel | numba |
| 100,000 | 0.012 (2.08) | 0.00287 (3.05) | numba-parallel | numba |
| 500,000 | 0.0843 (2.16) | 0.0142 (3.06) | numba-parallel | numba |
| 1,000,000 | 0.119 (2.2) | 0.0192 (3.07) | numba-parallel | numba |
| 3,000,000 | 0.507 (2.58) | 0.0727 (3.13) | numba-parallel | numba |
| 10,000,000 | 1.59 (3.67) | 0.242 (3.3) | numba-parallel | numba-parallel |

### P8 `mle_blocked`, obs sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 10,000 | 0.000918 (2.24) | 0.00081 (3.1) | numba-parallel | numba |
| 100,000 | 0.0109 (2.11) | 0.00226 (3.11) | numba-parallel | numba |
| 500,000 | 0.0772 (2.19) | 0.0118 (3.12) | numba-parallel | numba |
| 1,000,000 | 0.11 (2.22) | 0.0143 (3.12) | numba-parallel | numba |
| 3,000,000 | 0.465 (2.57) | 0.0587 (3.16) | numba-parallel | numba |
| 10,000,000 | 1.46 (3.56) | 0.192 (3.29) | numba-parallel | numba-parallel |

- NumPy's iterations in float32, against its float64 count: 18 against 17 at 3,000,000; 17 against 16 at 10,000,000
- 3,000,000, `numba`: 17 iterations, against NumPy's 18 in float32
- 3,000,000, `numba-parallel`: 17 iterations, against NumPy's 18 in float32
- 10,000,000, `numba`: 16 iterations, against NumPy's 17 in float32
- 10,000,000, `numba-parallel`: 16 iterations, against NumPy's 17 in float32

### P8 `mle_blocked`, params sweep, float64, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.00699 (2.18) | 0.00161 (3.04) | numba-parallel | numba |
| 10 | 0.0137 (2.08) | 0.0033 (3.06) | numba-parallel | numba |
| 30 | 0.0196 (2.09) | 0.00587 (3.05) | numba-parallel | numba |
| 100 | 0.0724 (2.14) | 0.0383 (3.08) | numba-parallel | numba |
| 300 | 0.459 (2.53) | 0.342 (3.37) | numba-parallel | numba |
| 1,000 | 6.77 (8.86) | 6.4 (9.28) | numba-parallel | numba |

### P8 `mle_blocked`, params sweep, float32, on qe-mini1

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---|---|
| 3 | 0.00652 (2.2) | 0.00128 (3.17) | numba-parallel | numba |
| 10 | 0.00798 (2.14) | 0.0019 (3.14) | numba-parallel | numba |
| 30 | 0.0112 (2.13) | 0.00294 (3.13) | numba-parallel | numba |
| 100 | 0.0274 (2.13) | 0.0137 (3.12) | numba-parallel | numba |
| 300 | 0.14 (2.24) | 0.0965 (3.2) | numba-parallel | numba |
| 1,000 | 1.98 (4.1) | 2.34 (5.46) | numba | numba |

