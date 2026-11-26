### P1 `vfi`, grid sweep, float64, on digitalocean-gpu-mi350x1-288gb-spot

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 100 | 0.0215 (1.39) | jax-gpu | jax-gpu |
| 200 | 0.0232 (1.4) | jax-gpu | jax-gpu |
| 400 | 0.0308 (1.37) | jax-gpu | jax-gpu |
| 700 | 0.0657 (1.39) | jax-gpu | jax-gpu |
| 1,000 | 0.111 (1.46) | jax-gpu | jax-gpu |
| 1,500 | 0.236 (1.57) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float64, on digitalocean-gpu-mi350x1-288gb-spot

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.373 (1.44) | jax-gpu | jax-gpu |
| 1,000 | 0.475 (1.57) | jax-gpu | jax-gpu |
| 5,000 | 0.549 (1.68) | jax-gpu | jax-gpu |
| 20,000 | 0.661 (1.76) | jax-gpu | jax-gpu |
| 50,000 | 1.02 (2.15) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on digitalocean-gpu-mi350x1-288gb-spot

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.355 (2.07) | jax-gpu | jax-gpu |
| 1,000 | 0.435 (2.18) | jax-gpu | jax-gpu |
| 5,000 | 0.527 (1.81) | jax-gpu | jax-gpu |
| 20,000 | 0.532 (1.93) | jax-gpu | jax-gpu |
| 50,000 | 0.986 (2.68) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: 2,212 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: 2,204 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: 2,212 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: 2,212 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: 2,212 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float64, on digitalocean-gpu-mi350x1-288gb-spot

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00509 (0.427) | jax-gpu | jax-gpu |
| 10,000 | 0.00506 (0.419) | jax-gpu | jax-gpu |
| 100,000 | 0.00561 (36) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0101 (0.61) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0598 (0.657) | jax-gpu | jax-gpu |

### P4 `wealth`, households sweep, float32, on digitalocean-gpu-mi350x1-288gb-spot

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00492 (0.348) | jax-gpu | jax-gpu |
| 10,000 | 0.00501 (0.342) | jax-gpu | jax-gpu |
| 100,000 | 0.00524 (0.358) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00807 (0.406) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0299 (0.504) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float64, on digitalocean-gpu-mi350x1-288gb-spot

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.274 (2.86) | jax-gpu | jax-gpu |
| 10 | 0.33 (2.86) | jax-gpu | jax-gpu |
| 100 | 0.301 (2.92) | jax-gpu | jax-gpu |
| 1,000 | 0.322 (2.87) | jax-gpu | jax-gpu |
| 10,000 | 0.584 (3.21) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on digitalocean-gpu-mi350x1-288gb-spot

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.242 (2.47) | jax-gpu | jax-gpu |
| 10 | 0.303 (2.99) | jax-gpu | jax-gpu |
| 100 | 0.264 (2.96) | jax-gpu | jax-gpu |
| 1,000 | 0.288 (3.02) | jax-gpu | jax-gpu |
| 10,000 | 0.494 (3.3) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float64, on digitalocean-gpu-mi350x1-288gb-spot

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.267 (2.82) | jax-gpu | jax-gpu |
| 5 | 0.273 (2.84) | jax-gpu | jax-gpu |
| 10 | 0.273 (2.85) | jax-gpu | jax-gpu |
| 20 | 0.274 (2.91) | jax-gpu | jax-gpu |
| 50 | 0.273 (3) | jax-gpu | jax-gpu |
| 100 | 0.273 (2.93) | jax-gpu | jax-gpu |
| 200 | 0.279 (2.93) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on digitalocean-gpu-mi350x1-288gb-spot

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.247 (20.1) | jax-gpu | jax-gpu |
| 5 | 0.245 (0.551) | jax-gpu | jax-gpu |
| 10 | 0.242 (2.48) | jax-gpu | jax-gpu |
| 20 | 0.261 (2.98) | jax-gpu | jax-gpu |
| 50 | 0.256 (3.17) | jax-gpu | jax-gpu |
| 100 | 0.26 (3.23) | jax-gpu | jax-gpu |
| 200 | 0.261 (5.27) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on digitalocean-gpu-mi350x1-288gb-spot (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.351 (1.95) | jax-gpu | jax-gpu |
| 1,000 | 0.425 (1.98) | jax-gpu | jax-gpu |
| 5,000 | 0.508 (1.8) | jax-gpu | jax-gpu |
| 20,000 | 0.544 (2.1) | jax-gpu | jax-gpu |
| 50,000 | 0.983 (2.69) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: 2,212 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: 2,204 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: 2,212 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: 2,212 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: 2,212 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float32, on digitalocean-gpu-mi350x1-288gb-spot (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00477 (0.341) | jax-gpu | jax-gpu |
| 10,000 | 0.00496 (4.38) | jax-gpu | jax-gpu |
| 100,000 | 0.00524 (0.368) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00793 (0.403) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0297 (0.508) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on digitalocean-gpu-mi350x1-288gb-spot (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.243 (2.49) | jax-gpu | jax-gpu |
| 10 | 0.305 (3.02) | jax-gpu | jax-gpu |
| 100 | 0.262 (3.02) | jax-gpu | jax-gpu |
| 1,000 | 0.286 (3.06) | jax-gpu | jax-gpu |
| 10,000 | 0.496 (3.45) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on digitalocean-gpu-mi350x1-288gb-spot (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.247 (0.545) | jax-gpu | jax-gpu |
| 5 | 0.245 (0.546) | jax-gpu | jax-gpu |
| 10 | 0.242 (2.47) | jax-gpu | jax-gpu |
| 20 | 0.263 (2.92) | jax-gpu | jax-gpu |
| 50 | 0.255 (3.24) | jax-gpu | jax-gpu |
| 100 | 0.261 (3.38) | jax-gpu | jax-gpu |
| 200 | 0.26 (5.01) | jax-gpu | jax-gpu |

