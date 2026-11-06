### P1 `vfi`, grid sweep, float64, on digitalocean-gpu-h200x1-141gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 100 | 0.0194 (0.775) | jax-gpu | jax-gpu |
| 200 | 0.0204 (1.16) | jax-gpu | jax-gpu |
| 400 | 0.0332 (1.18) | jax-gpu | jax-gpu |
| 700 | 0.0716 (1.22) | jax-gpu | jax-gpu |
| 1,000 | 0.136 (1.29) | jax-gpu | jax-gpu |
| 1,500 | 0.33 (1.61) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float64, on digitalocean-gpu-h200x1-141gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.317 (1.15) | jax-gpu | jax-gpu |
| 1,000 | 0.369 (1.21) | jax-gpu | jax-gpu |
| 5,000 | 0.466 (1.37) | jax-gpu | jax-gpu |
| 20,000 | 0.725 (1.56) | jax-gpu | jax-gpu |
| 50,000 | 0.977 (1.84) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on digitalocean-gpu-h200x1-141gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.106 (0.969) | jax-gpu | jax-gpu |
| 1,000 | 0.152 (1.03) | jax-gpu | jax-gpu |
| 5,000 | 0.154 (1.08) | jax-gpu | jax-gpu |
| 20,000 | 0.331 (1.27) | jax-gpu | jax-gpu |
| 50,000 | 0.506 (1.43) | jax-gpu | jax-gpu |

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

### P4 `wealth`, households sweep, float64, on digitalocean-gpu-h200x1-141gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00173 (0.716) | jax-gpu | jax-gpu |
| 10,000 | 0.00188 (0.758) | jax-gpu | jax-gpu |
| 100,000 | 0.00256 (0.75) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0114 (0.881) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0768 (0.925) | jax-gpu | jax-gpu |

### P4 `wealth`, households sweep, float32, on digitalocean-gpu-h200x1-141gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00151 (0.656) | jax-gpu | jax-gpu |
| 10,000 | 0.00143 (0.658) | jax-gpu | jax-gpu |
| 100,000 | 0.00155 (0.68) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00385 (0.788) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0253 (0.799) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float64, on digitalocean-gpu-h200x1-141gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.107 (0.9) | jax-gpu | jax-gpu |
| 10 | 0.0834 (0.892) | jax-gpu | jax-gpu |
| 100 | 0.0826 (0.872) | jax-gpu | jax-gpu |
| 1,000 | 0.0848 (0.896) | jax-gpu | jax-gpu |
| 10,000 | 0.185 (1.04) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on digitalocean-gpu-h200x1-141gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0724 (0.835) | jax-gpu | jax-gpu |
| 10 | 0.0772 (0.887) | jax-gpu | jax-gpu |
| 100 | 0.0753 (0.874) | jax-gpu | jax-gpu |
| 1,000 | 0.0759 (0.931) | jax-gpu | jax-gpu |
| 10,000 | 0.139 (0.991) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float64, on digitalocean-gpu-h200x1-141gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0864 (0.817) | jax-gpu | jax-gpu |
| 5 | 0.094 (0.81) | jax-gpu | jax-gpu |
| 10 | 0.105 (0.834) | jax-gpu | jax-gpu |
| 20 | 0.0873 (0.921) | jax-gpu | jax-gpu |
| 50 | 0.107 (1.2) | jax-gpu | jax-gpu |
| 100 | 0.07 (0.988) | jax-gpu | jax-gpu |
| 200 | 0.0767 (1.32) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on digitalocean-gpu-h200x1-141gb

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.077 (0.694) | jax-gpu | jax-gpu |
| 5 | 0.0782 (0.682) | jax-gpu | jax-gpu |
| 10 | 0.0834 (0.829) | jax-gpu | jax-gpu |
| 20 | 0.0932 (0.951) | jax-gpu | jax-gpu |
| 50 | 0.0836 (1.14) | jax-gpu | jax-gpu |
| 100 | 0.0712 (1.04) | jax-gpu | jax-gpu |
| 200 | 0.0704 (1.14) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on digitalocean-gpu-h200x1-141gb (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.287 (1.35) | jax-gpu | jax-gpu |
| 1,000 | 0.341 (1.37) | jax-gpu | jax-gpu |
| 5,000 | 0.33 (1.33) | jax-gpu | jax-gpu |
| 20,000 | 0.689 (1.95) | jax-gpu | jax-gpu |
| 50,000 | 0.822 (1.79) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: 2,211 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: 2,213 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: 2,209 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: 2,208 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: 2,214 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float32, on digitalocean-gpu-h200x1-141gb (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.0015 (0.627) | jax-gpu | jax-gpu |
| 10,000 | 0.0014 (0.633) | jax-gpu | jax-gpu |
| 100,000 | 0.00158 (0.683) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00393 (0.794) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0251 (0.902) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on digitalocean-gpu-h200x1-141gb (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0837 (0.856) | jax-gpu | jax-gpu |
| 10 | 0.0793 (0.919) | jax-gpu | jax-gpu |
| 100 | 0.0782 (0.913) | jax-gpu | jax-gpu |
| 1,000 | 0.0845 (1.06) | jax-gpu | jax-gpu |
| 10,000 | 0.177 (1.12) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on digitalocean-gpu-h200x1-141gb (highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0867 (0.709) | jax-gpu | jax-gpu |
| 5 | 0.0849 (0.7) | jax-gpu | jax-gpu |
| 10 | 0.0725 (0.869) | jax-gpu | jax-gpu |
| 20 | 0.0952 (1.09) | jax-gpu | jax-gpu |
| 50 | 0.081 (1.74) | jax-gpu | jax-gpu |
| 100 | 0.0703 (1.45) | jax-gpu | jax-gpu |
| 200 | 0.0807 (1.92) | jax-gpu | jax-gpu |

