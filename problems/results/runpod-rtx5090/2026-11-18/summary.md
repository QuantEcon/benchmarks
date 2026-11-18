### P1 `vfi`, grid sweep, float64, on runpod-rtx5090

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 100 | 0.0249 (2.84) | jax-gpu | jax-gpu |
| 200 | 0.032 (2.94) | jax-gpu | jax-gpu |
| 400 | 0.0744 (2.89) | jax-gpu | jax-gpu |
| 700 | 0.228 (3.13) | jax-gpu | jax-gpu |
| 1,000 | 0.406 (3.23) | jax-gpu | jax-gpu |
| 1,500 | 0.879 (3.94) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float64, on runpod-rtx5090

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.212 (0.993) | jax-gpu | jax-gpu |
| 1,000 | 0.241 (1.02) | jax-gpu | jax-gpu |
| 5,000 | 0.34 (1.13) | jax-gpu | jax-gpu |
| 20,000 | 0.706 (1.58) | jax-gpu | jax-gpu |
| 50,000 | 1.44 (2.3) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on runpod-rtx5090

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.0599 (0.739) | jax-gpu | jax-gpu |
| 1,000 | 0.0968 (0.783) | jax-gpu | jax-gpu |
| 5,000 | 0.126 (0.821) | jax-gpu | jax-gpu |
| 20,000 | 0.209 (0.899) | jax-gpu | jax-gpu |
| 50,000 | 0.389 (1.06) | jax-gpu | jax-gpu |

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

### P4 `wealth`, households sweep, float64, on runpod-rtx5090

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00215 (0.437) | jax-gpu | jax-gpu |
| 10,000 | 0.00214 (0.444) | jax-gpu | jax-gpu |
| 100,000 | 0.00833 (0.51) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0673 (0.692) | jax-gpu | jax-gpu |
| 10,000,000 | 0.633 (1.26) | jax-gpu | jax-gpu |

### P4 `wealth`, households sweep, float32, on runpod-rtx5090

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00106 (0.376) | jax-gpu | jax-gpu |
| 10,000 | 0.00115 (0.385) | jax-gpu | jax-gpu |
| 100,000 | 0.00129 (0.387) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00261 (0.481) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0133 (0.521) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float64, on runpod-rtx5090

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0538 (0.545) | jax-gpu | jax-gpu |
| 10 | 0.0454 (0.699) | jax-gpu | jax-gpu |
| 100 | 0.0475 (0.698) | jax-gpu | jax-gpu |
| 1,000 | 0.0927 (0.861) | jax-gpu | jax-gpu |
| 10,000 | 0.504 (1.26) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on runpod-rtx5090

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.041 (0.508) | jax-gpu | jax-gpu |
| 10 | 0.0454 (0.67) | jax-gpu | jax-gpu |
| 100 | 0.0451 (0.676) | jax-gpu | jax-gpu |
| 1,000 | 0.0437 (0.707) | jax-gpu | jax-gpu |
| 10,000 | 0.0857 (0.754) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float64, on runpod-rtx5090

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.05 (0.509) | jax-gpu | jax-gpu |
| 5 | 0.0536 (0.501) | jax-gpu | jax-gpu |
| 10 | 0.0542 (0.545) | jax-gpu | jax-gpu |
| 20 | 0.0477 (1.34) | jax-gpu | jax-gpu |
| 50 | 0.0558 (4.8) | jax-gpu | jax-gpu |
| 100 | 0.0639 (7.72) | jax-gpu | jax-gpu |
| 200 | 0.0927 (4) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on runpod-rtx5090

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0443 (0.415) | jax-gpu | jax-gpu |
| 5 | 0.0443 (0.404) | jax-gpu | jax-gpu |
| 10 | 0.0415 (0.523) | jax-gpu | jax-gpu |
| 20 | 0.0545 (1.08) | jax-gpu | jax-gpu |
| 50 | 0.0565 (2.46) | jax-gpu | jax-gpu |
| 100 | 0.056 (4.11) | jax-gpu | jax-gpu |
| 200 | 0.0496 (2.16) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on runpod-rtx5090 (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.161 (0.926) | jax-gpu | jax-gpu |
| 1,000 | 0.204 (0.965) | jax-gpu | jax-gpu |
| 5,000 | 0.253 (1.03) | jax-gpu | jax-gpu |
| 20,000 | 0.416 (1.19) | jax-gpu | jax-gpu |
| 50,000 | 0.638 (1.45) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: 2,211 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: 2,213 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: 2,209 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: 2,208 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: 2,214 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float32, on runpod-rtx5090 (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00109 (0.381) | jax-gpu | jax-gpu |
| 10,000 | 0.00109 (0.388) | jax-gpu | jax-gpu |
| 100,000 | 0.0013 (0.382) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00253 (0.467) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0133 (0.505) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on runpod-rtx5090 (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0414 (0.514) | jax-gpu | jax-gpu |
| 10 | 0.0446 (0.684) | jax-gpu | jax-gpu |
| 100 | 0.0445 (0.721) | jax-gpu | jax-gpu |
| 1,000 | 0.0442 (0.834) | jax-gpu | jax-gpu |
| 10,000 | 0.106 (0.863) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on runpod-rtx5090 (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0446 (0.427) | jax-gpu | jax-gpu |
| 5 | 0.0446 (0.404) | jax-gpu | jax-gpu |
| 10 | 0.0417 (0.504) | jax-gpu | jax-gpu |
| 20 | 0.0551 (1.32) | jax-gpu | jax-gpu |
| 50 | 0.0562 (4.6) | jax-gpu | jax-gpu |
| 100 | 0.0556 (8.85) | jax-gpu | jax-gpu |
| 200 | 0.0517 (3.75) | jax-gpu | jax-gpu |

