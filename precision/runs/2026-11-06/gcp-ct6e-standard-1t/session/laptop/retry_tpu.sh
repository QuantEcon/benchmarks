#!/bin/bash
# retry_tpu.sh <until HH:MM AEDT> <interval s> [zones]: try create.sh in each zone (default "us-east1-d us-east1-b") every <interval> s.
# Exits 0 once tpu-f64-stage1 exists, 2 on a quota error, 1 at the stop time.
U=$1; I=$2; ZONES=${3:-us-east1-d us-east1-b}
D=$HOME/work/quantecon/_runs/tpu-38
exists() { gcloud --configuration=qe-rental compute instances list --filter="name=tpu-f64-stage1 AND status!=PROVISIONING" --format="value(name,zone.basename(),status)" 2>/dev/null; }
while :; do
  [[ "$(TZ=Australia/Sydney date +%H:%M)" > "$U" ]] && { echo "stop time, no capacity"; exit 1; }
  for Z in $ZONES; do
    e=$(exists); [ -n "$e" ] && { echo "EXISTS: $e"; exit 0; }
    out=$(bash $D/create.sh $Z 2>&1); rc=$?
    echo "$out" >> $D/create-retries.log
    if [ $rc = 0 ]; then echo "CREATED in $Z at $(TZ=Australia/Sydney date +%H:%M:%S) AEDT"; echo "$out" | grep -v WARNING | tail -3; exit 0; fi
    echo "$(TZ=Australia/Sydney date +%H:%M) $Z: $(echo "$out" | grep -oE 'ZONE_RESOURCE_POOL_EXHAUSTED[A-Z_]*|Internal error|Quota [^ ]+ exceeded|stockout' | sort -u | tr '\n' ' ')"
    echo "$out" | grep -q 'Quota' && { echo "quota error: stopping"; exit 2; }
  done
  e=$(exists); [ -n "$e" ] && { echo "EXISTS: $e"; exit 0; }
  sleep $I
done
