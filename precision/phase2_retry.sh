#!/bin/bash
# Phase 2 of QuantEcon/benchmarks#1: keep offering RunsOn the runs it couldn't launch for lack of EC2 capacity.
#
#   bash precision/phase2_retry.sh <state file> [hours] [gap seconds]
#
# The state file has one line per instance still to run: "<instance> <devices> <run id or ->".
# RunsOn gives up on a job after about 45 minutes of launch attempts. It then either leaves the
# job queued on GitHub, or fails it from a small "fallback" instance at the "Set up runner" step.
# Every <gap> seconds (default 2 hours) this cancels each run still queued, and dispatches a fresh
# one in place of each queued or fallback-failed run. Every 5 minutes it checks whether a run has
# launched, meaning its checkout on the requested instance has started, and exits as soon as one
# has, so the session can watch it. A run that launched is never re-dispatched here, so a failing
# run can't be paid for twice. Gives up after <hours> (default 48).
R=QuantEcon/benchmarks
STATE=$1; HOURS=${2:-48}; GAP=${3:-7200}
END=$(( $(date +%s) + HOURS * 3600 ))
last=0

dispatch() {  # <instance> <devices>: prints the new run's id
  gh workflow run benchmark.yml -R $R --ref phase2-cloud-cards -f run_cpu=false -f run_precision=true \
    -f precision_run=full -f precision_instance="$1" -f precision_devices="$2" -f precision_extras=true \
    < /dev/null 2>&1 | grep -oE 'runs/[0-9]+' | cut -d/ -f2
}
# queued; launched (its checkout on the instance has started); fallback (failed before that step,
# which is how RunsOn fails a job it couldn't place); or starting (a runner is setting up)
job_state() {
  gh api repos/$R/actions/runs/$1/jobs --jq '.jobs[] | select(.name=="Precision Benchmarks") |
    if .status == "queued" then "queued"
    elif ([.steps[] | select(.name == "Checkout" and .status != "pending"
                             and .conclusion != "skipped")] | length) > 0 then "launched"
    elif .status == "completed" then "fallback"
    else "starting" end' < /dev/null 2>/dev/null
}

while [ "$(date +%s)" -lt "$END" ]; do
  while read -r inst dev id; do
    [ "$id" = - ] && continue
    if [ "$(job_state "$id")" = launched ]; then
      echo "$(date '+%F %T') $inst launched: run $id"
      exit 0
    fi
  done < "$STATE"

  if [ $(( $(date +%s) - last )) -ge "$GAP" ]; then
    : > "$STATE.new"; launched=""
    while read -r inst dev id; do
      if [ "$id" != - ]; then
        s=$(job_state "$id")       # again, so a run that launched a moment ago is never cancelled
        case "$s" in
          launched) echo "$inst $dev $id" >> "$STATE.new"; launched="$launched $inst ($id)"; continue ;;
          starting|"") echo "$inst $dev $id" >> "$STATE.new"; continue ;;   # look again next round
          queued) gh run cancel "$id" -R $R < /dev/null > /dev/null 2>&1; sleep 3 ;;
        esac                       # fallback: already failed, so just dispatch a fresh one
      fi
      new=$(dispatch "$inst" "$dev")
      echo "$inst $dev ${new:--}" >> "$STATE.new"
      echo "$(date '+%F %T') dispatched $inst: run ${new:-(failed)}" >> "$STATE.log"
    done < "$STATE"
    mv "$STATE.new" "$STATE"
    last=$(date +%s)
    [ -n "$launched" ] && { echo "$(date '+%F %T') launched:$launched"; exit 0; }
  fi
  sleep 300
done
echo "$(date '+%F %T') gave up after $HOURS hours: no capacity for $(cut -d' ' -f1 "$STATE" | tr '\n' ' ')"
