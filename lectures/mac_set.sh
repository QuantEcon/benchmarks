#!/bin/bash
# aiyagari_jax and ifp_egm_jax on qe-mini1's M6 CPU, both precisions, with the agents and the
# console's animated wallpaper paused (SIGSTOP) for the run and resumed (SIGCONT) after, even on failure.
cd ~/bench/lectures
OUT=mac
mkdir -p $OUT/aiyagari/as-used $OUT/aiyagari/scale $OUT/ifp/check $OUT/ifp/scale $OUT/ifp/as-used

state() {
  echo "== $1: $(date '+%F %T %Z')"; uptime
  ps -Ao pid,stat,pcpu,comm -r | head -10
  pmset -g therm | grep -v '^Note'; pmset -g | grep lowpowermode; pmset -g batt | head -1
}

# Each pattern is anchored to the start of its target's command line, so a shell
# whose own command merely mentions these names (one that launched this script,
# say) is not paused with them.
RC='^claude remote-control'
PAUSED="$(pgrep -f "$RC") $(pgrep -f '^[^ ]*/claude --print --sdk-url') \
  $(pgrep -f '^/System/Library/ExtensionKit/Extensions/WallpaperAerialsExtension') \
  $(pgrep -f '^/System/Library/Frameworks/VideoToolbox.framework/.*/VTDecoderXPCService')"
resume() {
  kill -CONT $PAUSED 2>/dev/null; echo "== resumed: $PAUSED at $(date '+%T')"
  # Stopping Remote Control's foreground job makes its tmux shell put it in the
  # background; give it the terminal back, but only if it has lost it. The
  # 2026-10-25 run did this step by hand.
  for p in $(pgrep -f "$RC"); do
    case "$(ps -o stat= -p $p)" in *+*) ;; *) tmux send-keys -t rc fg Enter ;; esac
  done
  state after >> $OUT/state.txt
}
trap resume EXIT

state before > $OUT/state.txt
echo "== pausing: $PAUSED" | tee -a $OUT/state.txt
kill -STOP $PAUSED
sleep 30
state paused >> $OUT/state.txt

CPU="env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg"
REC_A=(--record "K_star" "K_vals.tolist()" "np.asarray(σ_star).ravel().tolist()" "str(household.a_grid.dtype)")
REC_I=(--record "jax_time_without_compile" "numba_time_without_compile" "np.asarray(σ_star_egm_jax).ravel().tolist()" "np.asarray(σ_star_egm_nb).ravel().tolist()" "str(σ_star_egm_jax.dtype)")

for rep in 1 2 3; do for d in float64 float32; do
  $CPU .venv/bin/python run_lecture.py aiyagari_jax.md --dtype $d --out $OUT/aiyagari/as-used/cpu-$d-$rep.json --env-info ~/bench/benchmarks/precision "${REC_A[@]}"
  echo "$(date +%T) aiyagari as-used $d $rep (exit $?)"
done; done
for rep in 1 2 3; do for d in float64 float32; do
  $CPU .venv/bin/python aiyagari_scale.py aiyagari_jax.md --dtype $d --sizes 200,1000,2000,5000,10000 --out $OUT/aiyagari/scale/cpu-$d-$rep.json > /dev/null
  echo "$(date +%T) aiyagari scale $d $rep (exit $?)"
done; done
for rep in 1 2 3; do for d in float64 float32; do
  $CPU .venv-ifp/bin/python ifp_check.py ifp_egm_jax.md --dtype $d --sizes 200 --out $OUT/ifp/check/cpu-$d-$rep.json > /dev/null
  echo "$(date +%T) ifp check $d $rep (exit $?)"
done; done
for rep in 1 2 3; do for d in float64 float32; do
  $CPU .venv-ifp/bin/python ifp_scale.py ifp_egm_jax.md --dtype $d --sizes 200,1000,5000,20000,50000 --out $OUT/ifp/scale/cpu-$d-$rep.json > /dev/null
  echo "$(date +%T) ifp scale $d $rep (exit $?)"
done; done
for rep in 1 2 3; do
  $CPU .venv-ifp/bin/python run_lecture.py ifp_egm_jax.md --dtype float64 --out $OUT/ifp/as-used/cpu-float64-$rep.json --env-info ~/bench/benchmarks/precision "${REC_I[@]}"
  echo "$(date +%T) ifp as-used float64 $rep (exit $?)"
done
echo ALLDONE
