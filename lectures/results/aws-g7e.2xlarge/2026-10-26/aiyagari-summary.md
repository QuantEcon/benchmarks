repetitions: {('gpu', 'float32'): 1, ('gpu', 'float64'): 1}

### gpu: warm ms, median of runs; f64/f32 in brackets

| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 | get_value f64 · f32 | one HPI loop f64 · f32 |
|---|---|---|---|---|---|
| 200 | 0.08M | 0.13 · 0.10 (**1.3×**) | 0.07 · 0.07 (**1.0×**) | 1.62 · 1.56 (**1.0×**) | 1.9 · 1.9 (1.0×) |
| 1,000 | 2M | 0.23 · 0.13 (**1.8×**) | 0.11 · 0.06 (**1.8×**) | 2.02 · 1.44 (**1.4×**) | 2.5 · 2.0 (1.2×) |
| 2,000 | 8M | 0.42 · 0.13 (**3.3×**) | 0.15 · 0.09 (**1.7×**) | 2.06 · 1.39 (**1.5×**) | 2.6 · 1.9 (1.4×) |
| 5,000 | 50M | 1.74 · 0.17 (**9.9×**) | 0.45 · 0.22 (**2.1×**) | 2.60 · 1.37 (**1.9×**) | 4.6 · 2.0 (2.3×) |
| 10,000 | 200M | 6.43 · 0.33 (**19.6×**) | 1.66 · 0.62 (**2.7×**) | 1.97 · 1.42 (**1.4×**) | 8.8 · 2.1 (4.3×) |

### fidelity and convergence, every run

- gpu float64 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 15 it, policy 100.0% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 14 it, policy 100.0% as f64 run 1, v_sum -37689.6, stored==greedy True; 5000: 11 it, policy 100.0% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 12 it, policy 100.0% as f64 run 1, v_sum -188413, stored==greedy True
- gpu float32 run 1: 200: 13 it, policy 99.8% as f64 run 1, v_sum -3793.04, stored==greedy True; 1000: 22 it, policy 98.2% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 13 it, policy 83.5% as f64 run 1, v_sum -37689.8, stored==greedy True; 5000: 16 it, policy 61.1% as f64 run 1, v_sum -94211.4, stored==greedy True; 10000: 14 it, policy 46.6% as f64 run 1, v_sum -188415, stored==greedy True
