### P1 `vfi`, grid sweep, float64, on runpod-rtxpro6000-mig1g24

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 100 | 0.0548 (10.6) | jax-gpu | jax-gpu |
| 200 | 0.105 (10.1) | jax-gpu | jax-gpu |
| 400 | 0.295 (10.6) | jax-gpu | jax-gpu |
| 700 | 1.04 (11.3) | jax-gpu | jax-gpu |
| 1,000 | 1.89 (13.5) | jax-gpu | jax-gpu |
| 1,500 | 4.18 (13.6) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float64, on runpod-rtxpro6000-mig1g24

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.282 (2.05) | jax-gpu | jax-gpu |
| 1,000 | 0.38 (2.63) | jax-gpu | jax-gpu |
| 5,000 | 0.733 (2.72) | jax-gpu | jax-gpu |
| 20,000 | 2.32 (4.26) | jax-gpu | jax-gpu |
| 50,000 | 5.83 (7.96) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on runpod-rtxpro6000-mig1g24

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.0776 (1.9) | jax-gpu | jax-gpu |
| 1,000 | 0.117 (2.08) | jax-gpu | jax-gpu |
| 5,000 | 0.189 (2.02) | jax-gpu | jax-gpu |
| 20,000 | 0.402 (2.13) | jax-gpu | jax-gpu |
| 50,000 | 1.14 (2.87) | jax-gpu | jax-gpu |

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

### P4 `wealth`, households sweep, float64, on runpod-rtxpro6000-mig1g24

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00278 (1.01) | jax-gpu | jax-gpu |
| 10,000 | 0.00552 (1.04) | jax-gpu | jax-gpu |
| 100,000 | 0.0377 (1.34) | jax-gpu | jax-gpu |
| 1,000,000 | 0.309 (1.65) | jax-gpu | jax-gpu |
| 10,000,000 | 3.2 (4.58) | jax-gpu | jax-gpu |

### P4 `wealth`, households sweep, float32, on runpod-rtxpro6000-mig1g24

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00145 (0.947) | jax-gpu | jax-gpu |
| 10,000 | 0.0014 (0.917) | jax-gpu | jax-gpu |
| 100,000 | 0.00222 (2.87) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0072 (1.04) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0578 (1.13) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float64, on runpod-rtxpro6000-mig1g24

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.07 (1.49) | jax-gpu | jax-gpu |
| 10 | 0.0695 (2.59) | jax-gpu | jax-gpu |
| 100 | 0.0701 (2.68) | jax-gpu | jax-gpu |
| 1,000 | 0.279 (3.8) | jax-gpu | jax-gpu |
| 10,000 | 2.27 (5.55) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on runpod-rtxpro6000-mig1g24

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.062 (1.16) | jax-gpu | jax-gpu |
| 10 | 0.0657 (2.16) | jax-gpu | jax-gpu |
| 100 | 0.0638 (2.67) | jax-gpu | jax-gpu |
| 1,000 | 0.0643 (3.37) | jax-gpu | jax-gpu |
| 10,000 | 0.223 (3.68) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float64, on runpod-rtxpro6000-mig1g24

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0739 (1.23) | jax-gpu | jax-gpu |
| 5 | 0.0667 (1.29) | jax-gpu | jax-gpu |
| 10 | 0.0737 (1.55) | jax-gpu | jax-gpu |
| 20 | 0.0619 (8.89) | jax-gpu | jax-gpu |
| 50 | 0.0721 (28.2) | jax-gpu | jax-gpu |
| 100 | 0.0886 (38.1) | jax-gpu | jax-gpu |
| 200 | 0.243 (26.1) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on runpod-rtxpro6000-mig1g24

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0569 (0.862) | jax-gpu | jax-gpu |
| 5 | 0.0642 (0.884) | jax-gpu | jax-gpu |
| 10 | 0.0628 (1.32) | jax-gpu | jax-gpu |
| 20 | 0.0808 (7.3) | jax-gpu | jax-gpu |
| 50 | 0.0722 (19.9) | jax-gpu | jax-gpu |
| 100 | 0.074 (22.5) | jax-gpu | jax-gpu |
| 200 | 0.0701 (16.3) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on runpod-rtxpro6000-mig1g24 (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.237 (2.28) | jax-gpu | jax-gpu |
| 1,000 | 0.283 (2.28) | jax-gpu | jax-gpu |
| 5,000 | 0.431 (2.51) | jax-gpu | jax-gpu |
| 20,000 | 0.792 (2.61) | jax-gpu | jax-gpu |
| 50,000 | 1.87 (3.7) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: 2,211 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: 2,213 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: 2,209 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: 2,208 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: 2,214 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float32, on runpod-rtxpro6000-mig1g24 (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00141 (0.809) | jax-gpu | jax-gpu |
| 10,000 | 0.00142 (0.836) | jax-gpu | jax-gpu |
| 100,000 | 0.00228 (1.01) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00736 (1.04) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0578 (1.02) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on runpod-rtxpro6000-mig1g24 (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0542 (1.16) | jax-gpu | jax-gpu |
| 10 | 0.0571 (2.35) | jax-gpu | jax-gpu |
| 100 | 0.0706 (2.58) | jax-gpu | jax-gpu |
| 1,000 | 0.0744 (3.8) | jax-gpu | jax-gpu |
| 10,000 | 0.361 (4.19) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on runpod-rtxpro6000-mig1g24 (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0641 (0.893) | jax-gpu | jax-gpu |
| 5 | 0.0575 (0.911) | jax-gpu | jax-gpu |
| 10 | 0.0629 (1.21) | jax-gpu | jax-gpu |
| 20 | 0.0709 (10.9) | jax-gpu | jax-gpu |
| 50 | 0.0662 (27.6) | jax-gpu | jax-gpu |
| 100 | 0.0639 (45.1) | jax-gpu | jax-gpu |
| 200 | 0.0803 (24.2) | jax-gpu | jax-gpu |

