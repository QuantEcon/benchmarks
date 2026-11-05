### P1 `vfi`, grid sweep, float64, on aws-g7e.2xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 100 | 0.0325 (4.95) | jax-gpu | jax-gpu |
| 200 | 0.0397 (5.1) | jax-gpu | jax-gpu |
| 400 | 0.0837 (5.15) | jax-gpu | jax-gpu |
| 700 | 0.272 (5.39) | jax-gpu | jax-gpu |
| 1,000 | 0.48 (5.57) | jax-gpu | jax-gpu |
| 1,500 | 1.04 (6.02) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float64, on aws-g7e.2xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.339 (1.59) | jax-gpu | jax-gpu |
| 1,000 | 0.389 (1.75) | jax-gpu | jax-gpu |
| 5,000 | 0.529 (1.93) | jax-gpu | jax-gpu |
| 20,000 | 1.02 (2.38) | jax-gpu | jax-gpu |
| 50,000 | 1.65 (3.06) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on aws-g7e.2xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.0966 (1.17) | jax-gpu | jax-gpu |
| 1,000 | 0.145 (1.24) | jax-gpu | jax-gpu |
| 5,000 | 0.196 (1.31) | jax-gpu | jax-gpu |
| 20,000 | 0.316 (1.45) | jax-gpu | jax-gpu |
| 50,000 | 0.464 (1.62) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.12)
- 200, `jax-gpu`: 809 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.11)
- 1,000, `jax-gpu`: 1,005 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.11)
- 5,000, `jax-gpu`: 1,086 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.11)
- 20,000, `jax-gpu`: 1,117 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.11)
- 50,000, `jax-gpu`: 1,381 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float64, on aws-g7e.2xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00255 (0.539) | jax-gpu | jax-gpu |
| 10,000 | 0.00256 (0.571) | jax-gpu | jax-gpu |
| 100,000 | 0.0108 (0.676) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0808 (0.959) | jax-gpu | jax-gpu |
| 10,000,000 | 0.75 (1.64) | jax-gpu | jax-gpu |

### P4 `wealth`, households sweep, float32, on aws-g7e.2xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00132 (0.475) | jax-gpu | jax-gpu |
| 10,000 | 0.00133 (0.47) | jax-gpu | jax-gpu |
| 100,000 | 0.00133 (0.496) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00299 (0.623) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0149 (0.613) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float64, on aws-g7e.2xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0868 (0.883) | jax-gpu | jax-gpu |
| 10 | 0.0754 (1.14) | jax-gpu | jax-gpu |
| 100 | 0.0725 (1.19) | jax-gpu | jax-gpu |
| 1,000 | 0.118 (1.46) | jax-gpu | jax-gpu |
| 10,000 | 0.602 (1.91) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on aws-g7e.2xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0711 (0.798) | jax-gpu | jax-gpu |
| 10 | 0.0776 (1.05) | jax-gpu | jax-gpu |
| 100 | 0.0759 (1.08) | jax-gpu | jax-gpu |
| 1,000 | 0.0765 (1.21) | jax-gpu | jax-gpu |
| 10,000 | 0.103 (1.24) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float64, on aws-g7e.2xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0877 (0.812) | jax-gpu | jax-gpu |
| 5 | 0.0913 (0.775) | jax-gpu | jax-gpu |
| 10 | 0.0885 (0.887) | jax-gpu | jax-gpu |
| 20 | 0.0796 (2.52) | jax-gpu | jax-gpu |
| 50 | 0.0782 (10.2) | jax-gpu | jax-gpu |
| 100 | 0.0836 (17) | jax-gpu | jax-gpu |
| 200 | 0.111 (8.54) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on aws-g7e.2xlarge

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0741 (0.569) | jax-gpu | jax-gpu |
| 5 | 0.0752 (0.553) | jax-gpu | jax-gpu |
| 10 | 0.0709 (0.798) | jax-gpu | jax-gpu |
| 20 | 0.0944 (1.85) | jax-gpu | jax-gpu |
| 50 | 0.084 (5.04) | jax-gpu | jax-gpu |
| 100 | 0.0839 (8.64) | jax-gpu | jax-gpu |
| 200 | 0.0619 (4.37) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on aws-g7e.2xlarge (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.266 (1.7) | jax-gpu | jax-gpu |
| 1,000 | 0.318 (1.65) | jax-gpu | jax-gpu |
| 5,000 | 0.405 (1.83) | jax-gpu | jax-gpu |
| 20,000 | 0.645 (2.01) | jax-gpu | jax-gpu |
| 50,000 | 0.768 (2.14) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: 2,211 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: 2,213 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: 2,209 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: 2,208 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: 2,214 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float32, on aws-g7e.2xlarge (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00132 (0.473) | jax-gpu | jax-gpu |
| 10,000 | 0.00133 (0.485) | jax-gpu | jax-gpu |
| 100,000 | 0.00133 (0.488) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00296 (0.625) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0149 (0.63) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on aws-g7e.2xlarge (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0713 (0.809) | jax-gpu | jax-gpu |
| 10 | 0.0761 (1.2) | jax-gpu | jax-gpu |
| 100 | 0.0765 (1.18) | jax-gpu | jax-gpu |
| 1,000 | 0.0728 (1.43) | jax-gpu | jax-gpu |
| 10,000 | 0.125 (1.46) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on aws-g7e.2xlarge (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0752 (0.567) | jax-gpu | jax-gpu |
| 5 | 0.0746 (0.553) | jax-gpu | jax-gpu |
| 10 | 0.0716 (0.805) | jax-gpu | jax-gpu |
| 20 | 0.0951 (2.47) | jax-gpu | jax-gpu |
| 50 | 0.0837 (9.61) | jax-gpu | jax-gpu |
| 100 | 0.0846 (18.6) | jax-gpu | jax-gpu |
| 200 | 0.0657 (8.1) | jax-gpu | jax-gpu |

