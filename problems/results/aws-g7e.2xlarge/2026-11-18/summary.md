### P1 `vfi`, grid sweep, float64, on aws-g7e.2xlarge (pl300)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 100 | 0.0309 (4.88) | jax-gpu | jax-gpu |
| 200 | 0.0395 (5.18) | jax-gpu | jax-gpu |
| 400 | 0.0839 (5.24) | jax-gpu | jax-gpu |
| 700 | 0.274 (5.4) | jax-gpu | jax-gpu |
| 1,000 | 0.498 (5.41) | jax-gpu | jax-gpu |
| 1,500 | 1.12 (6.03) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float64, on aws-g7e.2xlarge (pl300)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.336 (1.65) | jax-gpu | jax-gpu |
| 1,000 | 0.39 (1.75) | jax-gpu | jax-gpu |
| 5,000 | 0.527 (1.92) | jax-gpu | jax-gpu |
| 20,000 | 1.02 (2.36) | jax-gpu | jax-gpu |
| 50,000 | 1.63 (3.02) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on aws-g7e.2xlarge (pl300)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.0968 (1.15) | jax-gpu | jax-gpu |
| 1,000 | 0.144 (1.24) | jax-gpu | jax-gpu |
| 5,000 | 0.198 (1.32) | jax-gpu | jax-gpu |
| 20,000 | 0.317 (1.48) | jax-gpu | jax-gpu |
| 50,000 | 0.464 (1.61) | jax-gpu | jax-gpu |

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

### P4 `wealth`, households sweep, float64, on aws-g7e.2xlarge (pl300)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00256 (0.541) | jax-gpu | jax-gpu |
| 10,000 | 0.00255 (0.571) | jax-gpu | jax-gpu |
| 100,000 | 0.0104 (0.68) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0794 (0.96) | jax-gpu | jax-gpu |
| 10,000,000 | 0.737 (1.61) | jax-gpu | jax-gpu |

### P4 `wealth`, households sweep, float32, on aws-g7e.2xlarge (pl300)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00133 (0.468) | jax-gpu | jax-gpu |
| 10,000 | 0.00133 (0.476) | jax-gpu | jax-gpu |
| 100,000 | 0.00132 (0.503) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00297 (0.626) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0173 (0.62) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float64, on aws-g7e.2xlarge (pl300)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0873 (0.872) | jax-gpu | jax-gpu |
| 10 | 0.0742 (1.15) | jax-gpu | jax-gpu |
| 100 | 0.0725 (1.18) | jax-gpu | jax-gpu |
| 1,000 | 0.115 (1.46) | jax-gpu | jax-gpu |
| 10,000 | 0.589 (1.91) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on aws-g7e.2xlarge (pl300)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0717 (0.814) | jax-gpu | jax-gpu |
| 10 | 0.0761 (1.04) | jax-gpu | jax-gpu |
| 100 | 0.0768 (1.07) | jax-gpu | jax-gpu |
| 1,000 | 0.0757 (1.23) | jax-gpu | jax-gpu |
| 10,000 | 0.103 (1.24) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float64, on aws-g7e.2xlarge (pl300)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0871 (0.799) | jax-gpu | jax-gpu |
| 5 | 0.0903 (0.774) | jax-gpu | jax-gpu |
| 10 | 0.0878 (0.915) | jax-gpu | jax-gpu |
| 20 | 0.079 (2.53) | jax-gpu | jax-gpu |
| 50 | 0.0785 (10.4) | jax-gpu | jax-gpu |
| 100 | 0.0802 (17) | jax-gpu | jax-gpu |
| 200 | 0.113 (8.55) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on aws-g7e.2xlarge (pl300)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0749 (0.566) | jax-gpu | jax-gpu |
| 5 | 0.0749 (0.558) | jax-gpu | jax-gpu |
| 10 | 0.0712 (0.781) | jax-gpu | jax-gpu |
| 20 | 0.0948 (1.83) | jax-gpu | jax-gpu |
| 50 | 0.0833 (5.05) | jax-gpu | jax-gpu |
| 100 | 0.083 (8.65) | jax-gpu | jax-gpu |
| 200 | 0.0609 (4.38) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on aws-g7e.2xlarge (pl300-highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.264 (1.6) | jax-gpu | jax-gpu |
| 1,000 | 0.32 (1.66) | jax-gpu | jax-gpu |
| 5,000 | 0.403 (1.77) | jax-gpu | jax-gpu |
| 20,000 | 0.649 (2.06) | jax-gpu | jax-gpu |
| 50,000 | 0.777 (2.17) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: 2,211 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: 2,213 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: 2,209 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: 2,208 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: 2,214 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float32, on aws-g7e.2xlarge (pl300-highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00132 (0.481) | jax-gpu | jax-gpu |
| 10,000 | 0.00132 (0.473) | jax-gpu | jax-gpu |
| 100,000 | 0.00133 (0.494) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00297 (0.628) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0173 (0.625) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on aws-g7e.2xlarge (pl300-highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0712 (0.857) | jax-gpu | jax-gpu |
| 10 | 0.0764 (1.15) | jax-gpu | jax-gpu |
| 100 | 0.0763 (1.2) | jax-gpu | jax-gpu |
| 1,000 | 0.0732 (1.49) | jax-gpu | jax-gpu |
| 10,000 | 0.126 (1.51) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on aws-g7e.2xlarge (pl300-highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0745 (0.56) | jax-gpu | jax-gpu |
| 5 | 0.0749 (0.558) | jax-gpu | jax-gpu |
| 10 | 0.0706 (0.801) | jax-gpu | jax-gpu |
| 20 | 0.095 (2.44) | jax-gpu | jax-gpu |
| 50 | 0.084 (9.67) | jax-gpu | jax-gpu |
| 100 | 0.0837 (18.6) | jax-gpu | jax-gpu |
| 200 | 0.0686 (8.04) | jax-gpu | jax-gpu |

