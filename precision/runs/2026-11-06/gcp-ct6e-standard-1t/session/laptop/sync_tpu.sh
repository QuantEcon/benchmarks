#!/bin/bash
# sync_tpu.sh <ip> <local dir> <stop HH:MM AEDT>: every 90 s, rsync ~/out and ~/session off the TPU VM, so a spot
# reclaim loses at most 90 s of results. Prints a line when the cell or lecture count changes. Exits after the
# final sync once ~/session/lectures.done exists, or at the stop time.
IP=$1; L=$2; U=$3
CM=$HOME/work/quantecon/_runs/tpu-38/cmt
SSH="ssh -o ControlPath=$CM -o BatchMode=yes -o ConnectTimeout=10"
mkdir -p "$L"
last=""
while :; do
  if rsync -a -e "$SSH" "ubuntu@$IP:out" "ubuntu@$IP:session" "$L/" 2> /dev/null; then
    c=$(wc -l < "$L/out/tpu-cells.jsonl" 2> /dev/null | tr -d ' '); l=$(grep -c 'rep' "$L/out/lectures/whole/lectures-set.log" 2> /dev/null)
    now="cells ${c:-0}, lectures ${l:-0}"
    [ "$now" != "$last" ] && echo "$(TZ=Australia/Sydney date +%H:%M:%S) synced: $now" && last=$now
    [ -f "$L/session/lectures.done" ] && { echo "lectures.done synced; exiting"; exit 0; }
  else
    echo "$(TZ=Australia/Sydney date +%H:%M:%S) sync failed (reclaimed or unreachable?)"
  fi
  [[ "$(TZ=Australia/Sydney date +%H:%M)" > "$U" ]] && { echo "stop time"; exit 0; }
  sleep 90
done
