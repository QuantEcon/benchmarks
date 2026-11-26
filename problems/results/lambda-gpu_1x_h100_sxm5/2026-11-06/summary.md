### P1 `vfi`, grid sweep, float64, on lambda-gpu_1x_h100_sxm5 (cuda12)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 100 | 0.0161 (0.785) | jax-gpu | jax-gpu |
| 200 | 0.0169 (1.31) | jax-gpu | jax-gpu |
| 400 | 0.0399 (1.35) | jax-gpu | jax-gpu |
| 700 | 0.0874 (1.37) | jax-gpu | jax-gpu |
| 1,000 | 0.162 (1.49) | jax-gpu | jax-gpu |
| 1,500 | 0.371 (1.83) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float64, on lambda-gpu_1x_h100_sxm5 (cuda12)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.304 (1.26) | jax-gpu | jax-gpu |
| 1,000 | 0.366 (1.34) | jax-gpu | jax-gpu |
| 5,000 | 0.466 (1.45) | jax-gpu | jax-gpu |
| 20,000 | 0.737 (1.74) | jax-gpu | jax-gpu |
| 50,000 | 1.11 (2.09) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on lambda-gpu_1x_h100_sxm5 (cuda12)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.102 (1.08) | jax-gpu | jax-gpu |
| 1,000 | 0.16 (1.13) | jax-gpu | jax-gpu |
| 5,000 | 0.213 (1.22) | jax-gpu | jax-gpu |
| 20,000 | 0.348 (1.4) | jax-gpu | jax-gpu |
| 50,000 | 0.588 (1.61) | jax-gpu | jax-gpu |

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

### P4 `wealth`, households sweep, float64, on lambda-gpu_1x_h100_sxm5 (cuda12)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00195 (0.764) | jax-gpu | jax-gpu |
| 10,000 | 0.00209 (0.785) | jax-gpu | jax-gpu |
| 100,000 | 0.00289 (0.816) | jax-gpu | jax-gpu |
| 1,000,000 | 0.0107 (1.02) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0819 (1.06) | jax-gpu | jax-gpu |

### P4 `wealth`, households sweep, float32, on lambda-gpu_1x_h100_sxm5 (cuda12)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00178 (0.66) | jax-gpu | jax-gpu |
| 10,000 | 0.0016 (0.661) | jax-gpu | jax-gpu |
| 100,000 | 0.00181 (0.681) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00414 (0.853) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0252 (0.861) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float64, on lambda-gpu_1x_h100_sxm5 (cuda12)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.1 (0.901) | jax-gpu | jax-gpu |
| 10 | 0.0793 (0.977) | jax-gpu | jax-gpu |
| 100 | 0.0793 (0.957) | jax-gpu | jax-gpu |
| 1,000 | 0.0838 (1.04) | jax-gpu | jax-gpu |
| 10,000 | 0.203 (1.15) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on lambda-gpu_1x_h100_sxm5 (cuda12)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0755 (0.874) | jax-gpu | jax-gpu |
| 10 | 0.0798 (0.98) | jax-gpu | jax-gpu |
| 100 | 0.0782 (1.01) | jax-gpu | jax-gpu |
| 1,000 | 0.0814 (1.06) | jax-gpu | jax-gpu |
| 10,000 | 0.147 (1.14) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float64, on lambda-gpu_1x_h100_sxm5 (cuda12)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.09 (0.852) | jax-gpu | jax-gpu |
| 5 | 0.0898 (0.867) | jax-gpu | jax-gpu |
| 10 | 0.102 (0.894) | jax-gpu | jax-gpu |
| 20 | 0.0822 (1.03) | jax-gpu | jax-gpu |
| 50 | 0.0867 (1.4) | jax-gpu | jax-gpu |
| 100 | 0.0802 (1.17) | jax-gpu | jax-gpu |
| 200 | 0.0857 (1.56) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on lambda-gpu_1x_h100_sxm5 (cuda12)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0808 (0.691) | jax-gpu | jax-gpu |
| 5 | 0.0792 (0.69) | jax-gpu | jax-gpu |
| 10 | 0.0744 (0.883) | jax-gpu | jax-gpu |
| 20 | 0.0989 (1.03) | jax-gpu | jax-gpu |
| 50 | 0.0874 (1.3) | jax-gpu | jax-gpu |
| 100 | 0.0814 (1.22) | jax-gpu | jax-gpu |
| 200 | 0.0838 (1.43) | jax-gpu | jax-gpu |

### P3 `egm`, grid sweep, float32, on lambda-gpu_1x_h100_sxm5 (cuda12-highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 200 | 0.279 (1.43) | jax-gpu | jax-gpu |
| 1,000 | 0.346 (1.48) | jax-gpu | jax-gpu |
| 5,000 | 0.437 (1.57) | jax-gpu | jax-gpu |
| 20,000 | 0.678 (2.1) | jax-gpu | jax-gpu |
| 50,000 | 0.967 (2.37) | jax-gpu | jax-gpu |

- 200, `jax-gpu`: 2,211 iterations, against NumPy's 2,192 in float64
- 1,000, `jax-gpu`: 2,213 iterations, against NumPy's 2,190 in float64
- 5,000, `jax-gpu`: 2,209 iterations, against NumPy's 2,190 in float64
- 20,000, `jax-gpu`: 2,208 iterations, against NumPy's 2,190 in float64
- 50,000, `jax-gpu`: 2,214 iterations, against NumPy's 2,190 in float64

### P4 `wealth`, households sweep, float32, on lambda-gpu_1x_h100_sxm5 (cuda12-highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,000 | 0.00181 (0.67) | jax-gpu | jax-gpu |
| 10,000 | 0.00163 (0.657) | jax-gpu | jax-gpu |
| 100,000 | 0.00185 (0.681) | jax-gpu | jax-gpu |
| 1,000,000 | 0.00409 (0.846) | jax-gpu | jax-gpu |
| 10,000,000 | 0.0252 (0.857) | jax-gpu | jax-gpu |

### P7 `kalman`, batch sweep, float32, on lambda-gpu_1x_h100_sxm5 (cuda12-highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1 | 0.0744 (0.888) | jax-gpu | jax-gpu |
| 10 | 0.0804 (1.06) | jax-gpu | jax-gpu |
| 100 | 0.0804 (1.01) | jax-gpu | jax-gpu |
| 1,000 | 0.0837 (1.2) | jax-gpu | jax-gpu |
| 10,000 | 0.19 (1.27) | jax-gpu | jax-gpu |

### P7 `kalman`, state sweep, float32, on lambda-gpu_1x_h100_sxm5 (cuda12-highest)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `jax-gpu` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 2 | 0.0788 (0.702) | jax-gpu | jax-gpu |
| 5 | 0.0807 (0.703) | jax-gpu | jax-gpu |
| 10 | 0.0751 (0.874) | jax-gpu | jax-gpu |
| 20 | 0.098 (1.26) | jax-gpu | jax-gpu |
| 50 | 0.0857 (2.11) | jax-gpu | jax-gpu |
| 100 | 0.0842 (1.67) | jax-gpu | jax-gpu |
| 200 | 0.0887 (2.46) | jax-gpu | jax-gpu |

