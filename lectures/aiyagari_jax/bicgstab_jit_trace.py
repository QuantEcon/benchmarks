"""
Run JAX's bicgstab body, compiled as JAX compiles it, with a print at each step, on the system where aiyagari_jax's float32 policy iteration goes NaN.

    python bicgstab_jit_trace.py aiyagari_jax.md --a-size 5000

Rebuilds the system from the lecture's own functions (loop 2 of Howard policy
iteration), then repeats jax.scipy.sparse.linalg.bicgstab's body step by step,
printing each scalar it divides by, until something is non-finite.
"""
import argparse
import re

CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)

parser = argparse.ArgumentParser()
parser.add_argument("lecture")
parser.add_argument("--a-size", type=int, default=5000)
args = parser.parse_args()
cells = [m.group(1) for m in CELL.finditer(open(args.lecture).read())]
last = next(i for i, c in enumerate(cells) if "def howard_policy_iteration" in c)
cells = [c.replace('jax.config.update("jax_enable_x64", True)', 'jax.config.update("jax_enable_x64", False)')
         for c in cells[:last + 1] if "!nvidia-smi" not in c]
ns = {}
for c in cells:
    exec(compile(c, "<lecture>", "exec"), ns)

import jax
import jax.numpy as jnp

hh, pr = ns["create_household"](a_size=args.a_size), ns["create_prices"]()
σ = jnp.zeros((args.a_size, 2), dtype=int)
v = ns["get_value"](σ, hh, pr)
σ = ns["get_greedy"](v, hh, pr)                      # the policy at loop 1
v = ns["get_value"](σ, hh, pr)
σ = ns["get_greedy"](v, hh, pr)                      # the policy at loop 2, where the NaN appears
b = ns["compute_r_σ"](σ, hh, pr)
A = jax.jit(lambda w: ns["R_σ"](w, σ, hh))

# JAX's _bicgstab_solve, as in jax/_src/scipy/sparse/linalg.py (0.12.0.1), plus jax.debug.print
def solve(b, maxiter=60, tol=1e-5):
    vdot = lambda u, w: jnp.vdot(u, w)
    atol2 = tol**2 * vdot(b, b)

    def cond_fun(value):
        x, r, *_, k = value
        return (vdot(r, r) > atol2) & (k < maxiter) & (k >= 0)

    def body_fun(value):
        x, r, rhat, alpha, omega, rho, p, q, k = value
        rho_ = vdot(rhat, r)
        beta = rho_ / rho * alpha / omega
        p_ = r + beta * (p - omega * q)
        q_ = A(p_)
        rq = vdot(rhat, q_)
        alpha_ = rho_ / rq
        s = r - alpha_ * q_
        exit_early = vdot(s, s) < atol2
        t = A(s)
        tt = vdot(t, t)
        omega_ = vdot(t, s) / tt
        x_ = jnp.where(exit_early, x + alpha_ * p_, x + alpha_ * p_ + omega_ * s)
        r_ = jnp.where(exit_early, s, s - omega_ * t)
        k_ = jnp.where((omega_ == 0) | (alpha_ == 0), -11, k + 1)
        k_ = jnp.where(rho_ == 0, -10, k_)
        jax.debug.print("{k:3d} |r|^2 {rs:11.4g}  rho_ {rho:11.4g}  <rhat,q_> {rq:11.4g}  alpha_ {a:11.4g}  "
                        "<t,t> {tt:11.4g}  omega_ {o:11.4g}  x finite {f}",
                        k=k, rs=vdot(r_, r_), rho=rho_, rq=rq, a=alpha_, tt=tt, o=omega_,
                        f=jnp.all(jnp.isfinite(x_)), ordered=True)
        return x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k_

    one = jnp.ones((), b.dtype)
    r0 = b - A(jnp.zeros_like(b))
    init = (jnp.zeros_like(b), r0, r0, one, one, one, r0, r0, 0)
    return jax.lax.while_loop(cond_fun, body_fun, init)

x, *_, k = jax.jit(solve)(b)
print(f"stopped with k = {int(k)}; x finite {bool(jnp.all(jnp.isfinite(x)))}")
