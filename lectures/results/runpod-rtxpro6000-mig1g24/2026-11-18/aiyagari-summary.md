repetitions: {('gpu', 'float32'): 1, ('gpu', 'float64'): 1}

### gpu: warm ms, median of runs; f64/f32 in brackets

| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 | get_value f64 · f32 | one HPI loop f64 · f32 |
|---|---|---|---|---|---|
| 200 | 0.08M | 0.30 · 0.32 (**0.9×**) | 0.18 · 0.20 (**0.9×**) | 1.96 · 1.76 (**1.1×**) | 2.9 · 3.0 (1.0×) |
| 1,000 | 2M | 0.54 · 0.23 (**2.3×**) | 0.26 · 0.22 (**1.2×**) | 2.35 · 1.60 (**1.5×**) | 3.4 · 2.5 (1.3×) |
| 2,000 | 8M | 1.41 · 0.31 (**4.6×**) | 0.51 · 0.18 (**2.9×**) | 2.36 · 1.81 (**1.3×**) | 4.1 · 2.5 (1.6×) |
| 5,000 | 50M | 6.93 · 0.46 (**15.2×**) | 1.66 · 0.83 (**2.0×**) | 2.85 · 1.78 (**1.6×**) | 10.1 · 2.9 (3.4×) |
| 10,000 | 200M | 26.34 · 1.03 (**25.6×**) | 6.67 · 2.40 (**2.8×**) | 2.17 · 1.79 (**1.2×**) | 29.1 · 3.1 (9.5×) |

### fidelity and convergence, every run

- gpu float64 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 15 it, policy 100.0% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 14 it, policy 100.0% as f64 run 1, v_sum -37689.6, stored==greedy True; 5000: 11 it, policy 100.0% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 12 it, policy 100.0% as f64 run 1, v_sum -188413, stored==greedy True
- gpu float32 run 1: 200: 13 it, policy 99.8% as f64 run 1, v_sum -3793.04, stored==greedy True; 1000: 22 it, policy 98.2% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 13 it, policy 83.5% as f64 run 1, v_sum -37689.8, stored==greedy True; 5000: 16 it, policy 61.1% as f64 run 1, v_sum -94211.4, stored==greedy True; 10000: 14 it, policy 46.6% as f64 run 1, v_sum -188415, stored==greedy True
