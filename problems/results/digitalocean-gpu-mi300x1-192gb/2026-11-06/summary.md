### P1 `vfi`, grid sweep, float64, on digitalocean-gpu-mi300x1-192gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 100 | 0.0217 (7.63) | jax-gpu | jax-gpu |
| 200 | 0.0236 (7.48) | jax-gpu | jax-gpu |
| 400 | 0.0358 (7.64) | jax-gpu | jax-gpu |
| 700 | 0.0831 (7.69) | jax-gpu | jax-gpu |
| 1,000 | 0.148 (7.89) | jax-gpu | jax-gpu |
| 1,500 | 0.328 (7.89) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float64, on digitalocean-gpu-mi300x1-192gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.639 (6.32) | jax-gpu | jax-gpu |
| 1,000 | 0.438 (6.25) | jax-gpu | jax-gpu |
| 5,000 | 0.591 (6.44) | jax-gpu | jax-gpu |
| 20,000 | 0.68 (6.47) | jax-gpu | jax-gpu |
| 50,000 | 1.38 (7.19) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on digitalocean-gpu-mi300x1-192gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.072 (3.27) | jax-gpu | jax-gpu |
| 1,000 | 0.112 (3.24) | jax-gpu | jax-gpu |
| 5,000 | 0.18 (1.99) | jax-gpu | jax-gpu |
| 20,000 | 0.274 (3.37) | jax-gpu | jax-gpu |
| 50,000 | 0.571 (4.02) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.26)
- 200, `jax-gpu`: 494 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.25)
- 1,000, `jax-gpu`: 629 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.25)
- 5,000, `jax-gpu`: 812 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.25)
- 20,000, `jax-gpu`: 878 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: answer differs from NumPy's in float64 (relative difference 0.25)
- 50,000, `jax-gpu`: 901 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float64, on digitalocean-gpu-mi300x1-192gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00501 (0.527) | jax-gpu | jax-gpu |
| 10,000 | 0.00506 (0.54) | jax-gpu | jax-gpu |
| 100,000 | 0.00575 (0.517) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0157 (0.773) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0824 (0.858) | jax-gpu | jax-gpu |

### P4 `wealth`, households sweep, float32, on digitalocean-gpu-mi300x1-192gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00472 (0.454) | jax-gpu | jax-gpu |
| 10,000 | 0.0047 (0.464) | jax-gpu | jax-gpu |
| 100,000 | 0.00521 (0.454) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0101 (0.519) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0393 (0.68) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float64, on digitalocean-gpu-mi300x1-192gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.155 (14.4) | jax-gpu | jax-gpu |
| 10 | 0.191 (14.5) | jax-gpu | jax-gpu |
| 100 | 0.16 (14.5) | jax-gpu | jax-gpu |
| 1,000 | 0.181 (14.3) | jax-gpu | jax-gpu |
| 10,000 | 0.479 (14.9) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on digitalocean-gpu-mi300x1-192gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.144 (5.68) | jax-gpu | jax-gpu |
| 10 | 0.156 (6.13) | jax-gpu | jax-gpu |
| 100 | 0.126 (6.31) | jax-gpu | jax-gpu |
| 1,000 | 0.143 (6.05) | jax-gpu | jax-gpu |
| 10,000 | 0.398 (6.72) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float64, on digitalocean-gpu-mi300x1-192gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.168 (14.5) | jax-gpu | jax-gpu |
| 5 | 0.174 (14.4) | jax-gpu | jax-gpu |
| 10 | 0.156 (14.4) | jax-gpu | jax-gpu |
| 20 | 0.156 (14.1) | jax-gpu | jax-gpu |
| 50 | 0.174 (14.5) | jax-gpu | jax-gpu |
| 100 | 0.188 (14.5) | jax-gpu | jax-gpu |
| 200 | 0.203 (14.5) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on digitalocean-gpu-mi300x1-192gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.146 (0.679) | jax-gpu | jax-gpu |
| 5 | 0.144 (0.7) | jax-gpu | jax-gpu |
| 10 | 0.141 (5.62) | jax-gpu | jax-gpu |
| 20 | 0.125 (6.04) | jax-gpu | jax-gpu |
| 50 | 0.141 (6.21) | jax-gpu | jax-gpu |
| 100 | 0.147 (6.09) | jax-gpu | jax-gpu |
| 200 | 0.173 (5.98) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on digitalocean-gpu-mi300x1-192gb (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.328 (3.31) | jax-gpu | jax-gpu |
| 1,000 | 0.38 (3.24) | jax-gpu | jax-gpu |
| 5,000 | 0.492 (2.33) | jax-gpu | jax-gpu |
| 20,000 | 0.68 (3.48) | jax-gpu | jax-gpu |
| 50,000 | 1.39 (4.34) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: 2,212 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: 2,204 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: 2,212 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: 2,212 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: 2,212 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float32, on digitalocean-gpu-mi300x1-192gb (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00471 (0.456) | jax-gpu | jax-gpu |
| 10,000 | 0.00469 (0.456) | jax-gpu | jax-gpu |
| 100,000 | 0.00507 (0.467) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0101 (0.531) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0387 (0.685) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on digitalocean-gpu-mi300x1-192gb (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.141 (5.7) | jax-gpu | jax-gpu |
| 10 | 0.154 (6.28) | jax-gpu | jax-gpu |
| 100 | 0.122 (6.23) | jax-gpu | jax-gpu |
| 1,000 | 0.14 (6.23) | jax-gpu | jax-gpu |
| 10,000 | 0.399 (6.95) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on digitalocean-gpu-mi300x1-192gb (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.145 (0.693) | jax-gpu | jax-gpu |
| 5 | 0.146 (0.723) | jax-gpu | jax-gpu |
| 10 | 0.145 (5.7) | jax-gpu | jax-gpu |
| 20 | 0.13 (5.94) | jax-gpu | jax-gpu |
| 50 | 0.142 (6.67) | jax-gpu | jax-gpu |
| 100 | 0.146 (6.69) | jax-gpu | jax-gpu |
| 200 | 0.174 (6.16) | jax-gpu | jax-gpu |

