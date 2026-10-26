repetitions: {('gpu', 'float32'): 1, ('gpu', 'float64'): 1}

### gpu: warm ms, median of runs; f64/f32 in brackets

| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 | get_value f64 · f32 | one HPI loop f64 · f32 |
|---|---|---|---|---|---|
| 200 | 0.08M | 0.26 · 0.28 (**0.9×**) | 0.17 · 0.19 (**0.9×**) | 2.74 · 2.95 (**0.9×**) | 3.5 · 3.7 (0.9×) |
| 1,000 | 2M | 0.31 · 0.29 (**1.1×**) | 0.18 · 0.19 (**0.9×**) | 2.90 · 2.61 (**1.1×**) | 3.9 · 3.8 (1.0×) |
| 2,000 | 8M | 0.35 · 0.32 (**1.1×**) | 0.26 · 0.20 (**1.3×**) | 2.89 · 2.91 (**1.0×**) | 3.9 · 4.0 (1.0×) |
| 5,000 | 50M | 0.67 · 0.52 (**1.3×**) | 0.53 · 0.35 (**1.5×**) | 2.77 · 2.86 (**1.0×**) | 4.0 · 4.0 (1.0×) |
| 10,000 | 200M | 1.97 · 1.09 (**1.8×**) | 1.38 · 0.79 (**1.7×**) | 2.41 · 2.93 (**0.8×**) | 5.8 · 4.7 (1.2×) |

### fidelity and convergence, every run

- gpu float64 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 15 it, policy 100.0% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 14 it, policy 100.0% as f64 run 1, v_sum -37689.6, stored==greedy True; 5000: 11 it, policy 100.0% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 12 it, policy 100.0% as f64 run 1, v_sum -188413, stored==greedy True
- gpu float32 run 1: 200: 13 it, policy 99.8% as f64 run 1, v_sum -3793.04, stored==greedy True; 1000: 22 it, policy 98.2% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 13 it, policy 83.5% as f64 run 1, v_sum -37689.8, stored==greedy True; 5000: 12 it, policy 60.2% as f64 run 1, v_sum -94211.3, stored==greedy True; 10000: 14 it, policy 46.6% as f64 run 1, v_sum -188415, stored==greedy True
