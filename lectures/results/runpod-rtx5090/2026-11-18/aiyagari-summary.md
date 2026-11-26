repetitions: {('gpu', 'float32'): 1, ('gpu', 'float64'): 1}

### gpu: warm ms, median of runs; f64/f32 in brackets

| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 | get_value f64 · f32 | one HPI loop f64 · f32 |
|---|---|---|---|---|---|
| 200 | 0.08M | 0.09 · 0.08 (**1.2×**) | 0.05 · 0.06 (**0.9×**) | 1.28 · 1.20 (**1.1×**) | 1.4 · 1.4 (1.0×) |
| 1,000 | 2M | 0.17 · 0.07 (**2.4×**) | 0.08 · 0.05 (**1.5×**) | 1.73 · 1.10 (**1.6×**) | 2.2 · 1.4 (1.6×) |
| 2,000 | 8M | 0.33 · 0.10 (**3.2×**) | 0.11 · 0.06 (**1.7×**) | 1.72 · 1.12 (**1.5×**) | 2.1 · 1.4 (1.4×) |
| 5,000 | 50M | 1.48 · 0.12 (**12.1×**) | 0.36 · 0.18 (**2.0×**) | 1.88 · 1.11 (**1.7×**) | 3.6 · 1.5 (2.3×) |
| 10,000 | 200M | 5.58 · 0.26 (**21.2×**) | 1.41 · 0.53 (**2.7×**) | 1.37 · 1.11 (**1.2×**) | 7.0 · 1.6 (4.3×) |

### fidelity and convergence, every run

- gpu float64 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 15 it, policy 100.0% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 14 it, policy 100.0% as f64 run 1, v_sum -37689.6, stored==greedy True; 5000: 11 it, policy 100.0% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 12 it, policy 100.0% as f64 run 1, v_sum -188413, stored==greedy True
- gpu float32 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.03, stored==greedy True; 1000: 15 it, policy 97.9% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 13 it, policy 83.5% as f64 run 1, v_sum -37689.8, stored==greedy True; 5000: 2 it, policy 0.3% as f64 run 1, v_sum nan NOT FINITE, stored==greedy True; 10000: 14 it, policy 49.6% as f64 run 1, v_sum -188415, stored==greedy True
