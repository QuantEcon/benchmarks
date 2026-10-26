repetitions: {('gpu', 'float32'): 1, ('gpu', 'float64'): 1}

### gpu: warm ms, median of runs; f64/f32 in brackets

| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 | get_value f64 · f32 | one HPI loop f64 · f32 |
|---|---|---|---|---|---|
| 200 | 0.08M | 0.18 · 0.15 (**1.2×**) | 0.11 · 0.10 (**1.0×**) | 2.14 · 1.79 (**1.2×**) | 2.5 · 2.4 (1.0×) |
| 1,000 | 2M | 0.72 · 0.15 (**4.7×**) | 0.18 · 0.13 (**1.4×**) | 2.50 · 1.70 (**1.5×**) | 3.2 · 2.4 (1.4×) |
| 2,000 | 8M | 1.43 · 0.21 (**6.6×**) | 0.37 · 0.18 (**2.0×**) | 2.58 · 1.68 (**1.5×**) | 4.0 · 2.3 (1.7×) |
| 5,000 | 50M | 6.27 · 0.40 (**15.7×**) | 1.60 · 0.82 (**1.9×**) | 3.57 · 1.97 (**1.8×**) | 9.9 · 2.8 (3.6×) |
| 10,000 | 200M | 23.23 · 1.14 (**20.4×**) | 6.14 · 2.04 (**3.0×**) | 2.41 · 1.84 (**1.3×**) | 26.3 · 3.2 (8.1×) |

### fidelity and convergence, every run

- gpu float64 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 15 it, policy 100.0% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 14 it, policy 100.0% as f64 run 1, v_sum -37689.6, stored==greedy True; 5000: 11 it, policy 100.0% as f64 run 1, v_sum -94210.9, stored==greedy True; 10000: 12 it, policy 100.0% as f64 run 1, v_sum -188413, stored==greedy True
- gpu float32 run 1: 200: 13 it, policy 100.0% as f64 run 1, v_sum -3793.05, stored==greedy True; 1000: 22 it, policy 98.2% as f64 run 1, v_sum -18850.7, stored==greedy True; 2000: 16 it, policy 90.1% as f64 run 1, v_sum -37689.7, stored==greedy True; 5000: 2 it, policy 0.3% as f64 run 1, v_sum nan NOT FINITE, stored==greedy True; 10000: 18 it, policy 50.2% as f64 run 1, v_sum -188414, stored==greedy True
