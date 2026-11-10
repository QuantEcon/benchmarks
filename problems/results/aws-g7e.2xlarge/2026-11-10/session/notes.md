# aws-g7e.2xlarge, QuantEcon/benchmarks#58 Kalman sweeps (project-benchmarks-jax#58)

- Instance: AWS g7e.2xlarge, us-east-1, i-01fce484ac30fe7c6 (name qe-kalman58-g7e, qe-rental tag), image ami-06d20c922ee616e37 "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261106"
- Launched 2026-11-10 20:47:24 UTC with aws-rental.sh (--profile qe-rental), booted 20:47:34 UTC, backstop shutdown 22:18:14 UTC. Run from the laptop session project-benchmarks-jax-a1.
- Commit: QuantEcon/benchmarks 314b0612e68943d0f2618b051e864ca859b5720d (main, includes #61 and #62)
- Card: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 595.112.07, power limit 600 W; driver line "driver 595.112.07: jax[cuda13]", label kalman-grid
- Check (--quick, SWEEPS=batch20,batch50,batch100): 20:48:28 to 20:49:55 UTC, 6 cells ok
- Full (SWEEPS=batch,state,batch20,batch50,batch100, TIMEOUT=1800): 20:50:19 to 21:05:27 UTC, 66 cells ok
- 2026-11-10.jsonl held 72 rows on the card: the check's 6 then the full run's 66. The committed `.jsonl` keeps the full run's 66 only, as the README says
- summary.md: no lines under any kalman-grid table
- Shell log: shell.log (tmux pipe-pane of session "run")
