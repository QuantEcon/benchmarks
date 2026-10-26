repetitions: {('gpu', 'float32'): 1, ('gpu', 'float64'): 1}

### gpu: warm ms, median of runs; f64/f32 in brackets

| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 | get_value f64 · f32 | one HPI loop f64 · f32 |
|---|---|---|---|---|---|
| 200 | 0.08M | 0.17 · 0.19 (**0.9×**) | 0.12 · 0.13 (**0.9×**) | 1.46 · 1.57 (**0.9×**) | 2.1 · 2.2 (1.0×) |
| 1,000 | 2M | 0.19 · 0.19 (**1.0×**) | 0.12 · 0.14 (**0.9×**) | 1.76 · 1.42 (**1.2×**) | 2.5 · 2.2 (1.1×) |
| 2,000 | 8M | 0.23 · 0.24 (**1.0×**) | 0.16 · 0.13 (**1.2×**) | 1.75 · 1.76 (**1.0×**) | 2.5 · 2.5 (1.0×) |
| 5,000 | 50M | 0.37 · 0.29 (**1.3×**) | 0.27 · 0.21 (**1.3×**) | 2.00 · 1.68 (**1.2×**) | 2.7 · 2.6 (1.1×) |
| 10,000 | 200M | 0.83 · 0.60 (**1.4×**) | 0.69 · 0.44 (**1.6×**) | 1.78 · 1.76 (**1.0×**) | 3.0 · 2.7 (1.1×) |

### fidelity and convergence, every run

- gpu float64 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 15 it, policy 100.0% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 14 it, policy 100.0% as f64 run 1, v_sum -37689.6, stored==greedy True; 5000: 11 it, policy 100.0% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 12 it, policy 100.0% as f64 run 1, v_sum -188413, stored==greedy True
- gpu float32 run 1: 200: 13 it, policy 99.8% as f64 run 1, v_sum -3793.04, stored==greedy True; 1000: 21 it, policy 97.2% as f64 run 1, v_sum -18850.8, stored==greedy True; 2000: 16 it, policy 90.1% as f64 run 1, v_sum -37689.7, stored==greedy True; 5000: 2 it, policy 0.3% as f64 run 1, v_sum nan NOT FINITE, stored==greedy True; 10000: 14 it, policy 46.8% as f64 run 1, v_sum -188415, stored==greedy True
