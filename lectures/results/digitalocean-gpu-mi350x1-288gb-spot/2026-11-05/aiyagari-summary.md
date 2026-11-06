repetitions: {('gpu', 'float32'): 1, ('gpu', 'float64'): 1}

### gpu: warm ms, median of runs; f64/f32 in brackets

| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 | get_value f64 · f32 | one HPI loop f64 · f32 |
|---|---|---|---|---|---|
| 200 | 0.08M | 0.34 · 0.32 (**1.1×**) | 0.17 · 0.16 (**1.1×**) | 2.38 · 2.36 (**1.0×**) | 3.1 · 3.0 (1.0×) |
| 1,000 | 2M | 0.33 · 0.32 (**1.0×**) | 0.16 · 0.16 (**1.0×**) | 2.42 · 2.29 (**1.1×**) | 3.2 · 3.3 (1.0×) |
| 2,000 | 8M | 0.34 · 0.33 (**1.0×**) | 0.16 · 0.16 (**1.0×**) | 2.64 · 2.42 (**1.1×**) | 3.5 · 3.4 (1.0×) |
| 5,000 | 50M | 0.37 · 0.33 (**1.1×**) | 0.25 · 0.16 (**1.5×**) | 2.77 · 2.50 (**1.1×**) | 3.8 · 3.5 (1.1×) |
| 10,000 | 200M | 0.95 · 0.39 (**2.4×**) | 0.53 · 0.31 (**1.7×**) | 2.64 · 2.59 (**1.0×**) | 4.4 · 3.8 (1.1×) |

### fidelity and convergence, every run

- gpu float64 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 15 it, policy 100.0% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 14 it, policy 100.0% as f64 run 1, v_sum -37689.6, stored==greedy True; 5000: 11 it, policy 100.0% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 12 it, policy 100.0% as f64 run 1, v_sum -188413, stored==greedy True
- gpu float32 run 1: 200: 13 it, policy 99.8% as f64 run 1, v_sum -3793.03, stored==greedy True; 1000: 14 it, policy 98.5% as f64 run 1, v_sum -18850.8, stored==greedy True; 2000: 14 it, policy 94.4% as f64 run 1, v_sum -37689.4, stored==greedy True; 5000: 16 it, policy 62.8% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 14 it, policy 46.5% as f64 run 1, v_sum -188414, stored==greedy True
