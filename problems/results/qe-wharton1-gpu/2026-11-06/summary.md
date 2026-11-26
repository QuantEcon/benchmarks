### P1 `vfi`, grid sweep, float64, on qe-wharton1-gpu (numa-default)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---|---|
| 1,500 | 145 (141) | 33.5 (37.6) | 35.4 (34.9) | numba-parallel | jax-cpu |

### P3 `egm`, grid sweep, float64, on qe-wharton1-gpu (numa-default)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---|---|
| 50,000 | 289 (296) | 306 (315) | 301 (315) | numpy | numpy |

### P4 `wealth`, households sweep, float64, on qe-wharton1-gpu (numa-default)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---|---|
| 10,000,000 | 128 (128) | 5.15 (6.95) | 85.4 (84.1) | numba-parallel | numba-parallel |

### P7 `kalman`, batch sweep, float64, on qe-wharton1-gpu (numa-default)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 10,000 | 47.8 (46.8) | 66.5 (78.9) | 197 (215) | 63.8 (63.6) | numpy | numpy |

### P1 `vfi`, grid sweep, float64, on qe-wharton1-gpu (numa-memory-nodes)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---|---|
| 1,500 | 139 (141) | 42.2 (46.3) | 34.1 (27.8) | jax-cpu | jax-cpu |

### P3 `egm`, grid sweep, float64, on qe-wharton1-gpu (numa-memory-nodes)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---|---|
| 50,000 | 273 (282) | 298 (304) | 274 (285) | numpy | numpy |

### P4 `wealth`, households sweep, float64, on qe-wharton1-gpu (numa-memory-nodes)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---|---|
| 10,000,000 | 128 (128) | 10.3 (12) | 82.7 (84.1) | numba-parallel | numba-parallel |

### P7 `kalman`, batch sweep, float64, on qe-wharton1-gpu (numa-memory-nodes)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 10,000 | 43.6 (42.4) | 63.9 (76) | 157 (174) | 54.3 (55) | numpy | numpy |

### P1 `vfi`, grid sweep, float64, on qe-wharton1-gpu (numa-one-node)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---|---|
| 1,500 | 138 (141) | 48.3 (51.7) | 34.4 (34.5) | jax-cpu | jax-cpu |

### P3 `egm`, grid sweep, float64, on qe-wharton1-gpu (numa-one-node)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---|---|
| 50,000 | 266 (267) | 288 (295) | 257 (265) | numba-parallel | numba-parallel |

### P4 `wealth`, households sweep, float64, on qe-wharton1-gpu (numa-one-node)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---|---|
| 10,000,000 | 128 (128) | 20.6 (22.3) | 93.1 (93.7) | numba-parallel | numba-parallel |

### P7 `kalman`, batch sweep, float64, on qe-wharton1-gpu (numa-one-node)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | `numba` | `numba-parallel` | `jax-cpu` | Fastest warm | Fastest cold |
|---:|---:|---:|---:|---:|---|---|
| 10,000 | 41.4 (40.3) | 63.1 (75.3) | 89 (106) | 42.6 (43.1) | numpy | numpy |

### P1 `vfi`, grid sweep, float64, on qe-wharton1-gpu (numa-remote-memory)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numba` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 1,500 | 163 (166) | numba | numba |

### P3 `egm`, grid sweep, float64, on qe-wharton1-gpu (numa-remote-memory)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 50,000 | 312 (315) | numpy | numpy |

### P4 `wealth`, households sweep, float64, on qe-wharton1-gpu (numa-remote-memory)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 10,000,000 | 142 (142) | numpy | numpy |

### P7 `kalman`, batch sweep, float64, on qe-wharton1-gpu (numa-remote-memory)

Median warm time, with cold time (compile and first run) in brackets, in seconds.

| Size | `numpy` | Fastest warm | Fastest cold |
|---:|---:|---|---|
| 10,000 | 45.6 (43.4) | numpy | numpy |
