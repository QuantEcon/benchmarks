repetitions: {('gpu', 'float32'): 1, ('gpu', 'float64'): 1}

### gpu: warm ms, median of runs; f64/f32 in brackets

| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 | get_value f64 · f32 | one HPI loop f64 · f32 |
|---|---|---|---|---|---|
| 200 | 0.08M | 0.13 · 0.12 (**1.0×**) | 0.09 · 0.09 (**1.0×**) | 1.80 · 1.84 (**1.0×**) | 2.1 · 2.2 (1.0×) |
| 1,000 | 2M | 0.19 · 0.14 (**1.4×**) | 0.14 · 0.09 (**1.5×**) | 1.67 · 1.66 (**1.0×**) | 2.1 · 2.4 (0.9×) |
| 2,000 | 8M | 0.22 · 0.20 (**1.1×**) | 0.19 · 0.15 (**1.3×**) | 1.63 · 1.63 (**1.0×**) | 2.1 · 2.4 (0.9×) |
| 5,000 | 50M | 0.41 · 0.31 (**1.3×**) | 0.25 · 0.20 (**1.2×**) | 1.92 · 1.56 (**1.2×**) | 2.5 · 2.8 (0.9×) |
| 10,000 | 200M | 0.82 · 0.57 (**1.4×**) | 0.60 · 0.36 (**1.7×**) | 1.73 · 1.66 (**1.0×**) | 2.8 · 2.7 (1.0×) |

### fidelity and convergence, every run

- gpu float64 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 15 it, policy 100.0% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 14 it, policy 100.0% as f64 run 1, v_sum -37689.6, stored==greedy True; 5000: 11 it, policy 100.0% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 12 it, policy 100.0% as f64 run 1, v_sum -188413, stored==greedy True
- gpu float32 run 1: 200: 13 it, policy 99.8% as f64 run 1, v_sum -3793.04, stored==greedy True; 1000: 21 it, policy 97.2% as f64 run 1, v_sum -18850.8, stored==greedy True; 2000: 13 it, policy 83.5% as f64 run 1, v_sum -37689.8, stored==greedy True; 5000: 2 it, policy 0.3% as f64 run 1, v_sum nan NOT FINITE, stored==greedy True; 10000: 14 it, policy 46.6% as f64 run 1, v_sum -188415, stored==greedy True
