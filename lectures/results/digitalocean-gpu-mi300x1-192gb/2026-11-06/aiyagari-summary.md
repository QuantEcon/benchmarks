repetitions: {('gpu', 'float32'): 1, ('gpu', 'float64'): 1}

### gpu: warm ms, median of runs; f64/f32 in brackets

| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 | get_value f64 · f32 | one HPI loop f64 · f32 |
|---|---|---|---|---|---|
| 200 | 0.08M | 0.33 · 0.33 (**1.0×**) | 0.17 · 0.17 (**1.0×**) | 2.46 · 2.33 (**1.1×**) | 3.2 · 3.2 (1.0×) |
| 1,000 | 2M | 0.30 · 0.29 (**1.0×**) | 0.17 · 0.17 (**1.0×**) | 2.39 · 2.30 (**1.0×**) | 3.3 · 3.4 (1.0×) |
| 2,000 | 8M | 0.34 · 0.28 (**1.2×**) | 0.18 · 0.17 (**1.0×**) | 2.77 · 2.38 (**1.2×**) | 3.6 · 3.4 (1.1×) |
| 5,000 | 50M | 0.45 · 0.34 (**1.3×**) | 0.24 · 0.17 (**1.4×**) | 2.91 · 2.49 (**1.2×**) | 4.0 · 3.7 (1.1×) |
| 10,000 | 200M | 1.22 · 0.43 (**2.9×**) | 0.77 · 0.42 (**1.8×**) | 2.54 · 2.54 (**1.0×**) | 4.5 · 3.8 (1.2×) |

### fidelity and convergence, every run

- gpu float64 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 15 it, policy 100.0% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 14 it, policy 100.0% as f64 run 1, v_sum -37689.6, stored==greedy True; 5000: 11 it, policy 100.0% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 12 it, policy 100.0% as f64 run 1, v_sum -188413, stored==greedy True
- gpu float32 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.06, stored==greedy True; 1000: 14 it, policy 98.5% as f64 run 1, v_sum -18850.8, stored==greedy True; 2000: 14 it, policy 94.4% as f64 run 1, v_sum -37689.4, stored==greedy True; 5000: 16 it, policy 62.8% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 14 it, policy 46.5% as f64 run 1, v_sum -188414, stored==greedy True
