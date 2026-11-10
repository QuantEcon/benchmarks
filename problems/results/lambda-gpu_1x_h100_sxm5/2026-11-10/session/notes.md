# lambda-gpu_1x_h100_sxm5, QuantEcon/benchmarks#58 Kalman sweeps (project-benchmarks-jax#58)

- Instance: Lambda gpu_1x_h100_sxm5, us-southeast-1, id 5848811e0e1d458b8131bc3cfd30d44c, Lambda default image (Ubuntu 24.04.4 LTS)
- Launched 2026-11-10 20:43:51 UTC, active 20:45:39 UTC. Run from the laptop session project-benchmarks-jax-a1.
- Commit: QuantEcon/benchmarks 314b0612e68943d0f2618b051e864ca859b5720d (main, includes #61 and #62)
- Card: NVIDIA H100 80GB HBM3, driver 580.105.08, power limit 700 W; driver line "driver 580.105.08: jax[cuda13]", label kalman-grid
- Check (--quick, SWEEPS=batch20,batch50,batch100): 20:45:58 to 20:46:21 UTC, 6 cells ok
- Full (SWEEPS=batch,state,batch20,batch50,batch100, TIMEOUT=1800): 20:46:57 to 20:51:15 UTC, 66 cells ok
- 2026-11-10.jsonl holds 72 rows: the check's 6 then the full run's 66
- summary.md: no lines under any kalman-grid table
- Shell log: shell.log (tmux pipe-pane of session "run")
