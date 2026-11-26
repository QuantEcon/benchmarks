#!/usr/bin/env bash
# Why float32 changes what seven lecture-jax lectures print: each check swaps one ingredient between the precisions.
#
#   ./run_checks.sh <lecture-jax's lectures/ at 4b939ec> [python] > float32.txt
#
# Needs jax 0.12.0.1, quantecon and numba. Runs on the CPU; these are diagnostic runs, not measurements.
set -euo pipefail
L=$1
PY=${2:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
cd "$OUT"
run() { "$PY" -W ignore "$HERE/$1" "${@:2}"; }

echo "## mle_jax, cell #19: Newton-Raphson, with x and y drawn in one precision and solved in another"
run mle_check.py gen 64 d64.npz
run mle_check.py gen 32 d32.npz
for data in 64 32; do for comp in 64 32; do
    echo "data drawn in float$data, solved in float$comp:"; run mle_check.py solve $comp d$data.npz
done; done

echo; echo "## hopenhayn, cell #26: the equilibrium price p*, with the Monte Carlo draws made in one precision and the model solved in another"
run hopenhayn_check.py gen 64 h64.npz "$L/hopenhayn.md"
run hopenhayn_check.py gen 32 h32.npz "$L/hopenhayn.md"
for data in 64 32; do for comp in 64 32; do
    echo "draws in float$data, solved in float$comp:"; run hopenhayn_check.py solve $comp h$data.npz "$L/hopenhayn.md"
done; done

echo; echo "## markov_asset_jax, cell #15: the 15-step power-iteration estimate of the spectral radius, from each precision's start vector"
run markov_asset_check.py gen 64 b64.npy "$L/markov_asset_jax.md"
run markov_asset_check.py gen 32 b32.npy "$L/markov_asset_jax.md"
for s in 64 32; do for c in 64 32; do
    echo "start vector drawn in float$s, iterated in float$c:"; run markov_asset_check.py run $c b$s.npy "$L/markov_asset_jax.md"
done; done

echo; echo "## arellano_jax: the solution in float32 against float64, and float64 at other tolerances against its own at the lecture's 1e-8"
run arellano_check.py 64 "$L/arellano_jax.md" a64.npz
run arellano_check.py 32 "$L/arellano_jax.md" a32.npz
echo "float32 against float64, both at the lecture's tol = 1e-8:"; run arellano_check.py compare a64.npz a32.npz
for t in 1e-5 1e-6 1e-10; do
    run arellano_check.py 64 "$L/arellano_jax.md" a64_$t.npz $t
    echo "float64 at tol = $t against float64 at 1e-8:"; run arellano_check.py compare a64.npz a64_$t.npz
done

for lec in "opt_invest 18 19" "opt_savings_2 22 23"; do
    set -- $lec
    echo; echo "## $1: Howard policy iteration (HPI), VFI and OPI with m = 10, in each precision, against float64's HPI"
    run policy_check.py 64 "$L/$1.md" $2 $3 p64.npz
    run policy_check.py 32 "$L/$1.md" $2 $3 p32.npz
    run policy_check.py compare p64.npz p32.npz
done

echo; echo "## ifp_egm_jax, cells #11 and #14: one application of the lecture's Numba operator K_egm_nb, on float64 and float32 arrays"
run ifp_numba_check.py "$L/ifp_egm_jax.md"
