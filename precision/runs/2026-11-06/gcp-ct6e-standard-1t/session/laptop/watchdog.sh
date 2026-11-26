#!/bin/bash
# watchdog.sh <name> <local dir> <give up HH:MM AEDT> [minutes, default 55]: wait for instance <name> to exist; <minutes> after its creation
# time, copy ~/out and ~/session off (best effort), delete it in its zone, and list until it is gone.
# Keeps watching until the give-up time: a failed create can list the instance briefly as PROVISIONING, then drop it.
set -u
N=$1; L=$2; U=$3; M=${4:-55}
G="gcloud --configuration=qe-rental"
CM=$HOME/work/quantecon/_runs/tpu-38/cmt
seen=0
while :; do
  row=$($G compute instances list --filter="name=$N" --format="csv[no-heading](zone.basename(),creationTimestamp,networkInterfaces[0].accessConfigs[0].natIP,status)" 2>/dev/null)
  now=$(date +%s)
  if [ -z "$row" ]; then
    [ $seen = 1 ] && { echo "$(TZ=Australia/Sydney date +%H:%M:%S) $N is gone; still watching"; seen=0; }
    [[ "$(TZ=Australia/Sydney date +%H:%M)" > "$U" ]] && { echo "give-up time; $N not listed"; exit 0; }
  else
    IFS=, read -r Z CT IP ST <<< "$row"
    DL=$(( $(python3 -I -c "import sys, datetime; print(int(datetime.datetime.fromisoformat(sys.argv[1]).timestamp()))" "$CT") + M * 60 ))
    [ $seen = 0 ] && echo "$(TZ=Australia/Sydney date +%H:%M:%S) saw $N in $Z ($ST, $IP), created $CT; delete due $(TZ=Australia/Sydney date -r $DL +%H:%M:%S) AEDT"
    seen=1
    if [ $now -ge $DL ]; then
      echo "$(TZ=Australia/Sydney date +%H:%M:%S) deadline: copying, then deleting $N"
      mkdir -p "$L/watchdog-copy"
      scp -o ControlPath="$CM" -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15 -q -r \
        "ubuntu@$IP:out" "ubuntu@$IP:session" "$L/watchdog-copy/" && echo copied || echo "copy failed"
      $G compute instances delete "$N" --zone="$Z" --quiet 2>&1 | grep -v WARNING
      for i in $(seq 1 40); do
        $G compute instances list --filter="name=$N" --format="value(name)" 2>/dev/null | grep -qx "$N" || { echo "gone at $(TZ=Australia/Sydney date +%H:%M:%S) AEDT"; break; }
        sleep 15
      done
      seen=0
    fi
  fi
  sleep 60
done
