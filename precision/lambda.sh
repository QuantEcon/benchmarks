#!/bin/bash
# Phase 2 of #1: rent a GPU from Lambda's cloud, through its API (https://docs.lambda.ai/api/cloud).
#
#   bash precision/lambda.sh capacity [pattern]     instance types matching pattern, their price, and where they're free
#   bash precision/lambda.sh wait <type> [seconds]  poll until <type> has capacity somewhere; prints the region
#   bash precision/lambda.sh add-key [public key]   register an SSH key as qe-benchmarks (default ~/.ssh/id_ed25519.pub)
#   bash precision/lambda.sh launch <type> <region> launch one instance with that key; prints its id
#   bash precision/lambda.sh list                   running instances: id, type, region, status, IP
#   bash precision/lambda.sh terminate <id>         stop paying for an instance
#
# The API key comes from LAMBDA_API_KEY, or else ~/.config/lambda/api_key (chmod 600). It is never printed.
set -euo pipefail
API=https://cloud.lambda.ai/api/v1
KEY_NAME=qe-benchmarks
KEY=${LAMBDA_API_KEY:-$(cat "$HOME/.config/lambda/api_key" 2>/dev/null || true)}

call() {  # <method> <path> [json body]: prints the response's data, or its error and exits
  [ -n "$KEY" ] || { echo "no API key: set LAMBDA_API_KEY or write it to ~/.config/lambda/api_key" >&2; exit 2; }
  local out
  out=$(curl -sS -X "$1" "$API$2" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        ${3:+--data "$3"})
  python3 -c '
import json, sys
r = json.loads(sys.argv[1])
if "error" in r:
    e = r["error"]
    sys.exit("Lambda API error %s: %s %s" % (e.get("code"), e.get("message"), e.get("suggestion") or ""))
print(json.dumps(r["data"]))' "$out"
}

case "${1:-}" in
  capacity)
    data=$(call GET /instance-types)
    python3 -c '
import json, sys
pattern = sys.argv[1].lower()
for name, t in sorted(json.load(sys.stdin).items()):
    it = t["instance_type"]
    if pattern in (name + " " + it["gpu_description"]).lower():
        regions = ", ".join(r["name"] for r in t["regions_with_capacity_available"]) or "none free"
        print("%-28s %-32s USD %6.2f/hr  %s" % (name, it["gpu_description"], it["price_cents_per_hour"] / 100, regions))' "${2:-}" <<< "$data"
    ;;
  wait)
    while true; do
      data=$(call GET /instance-types)
      region=$(python3 -c '
import json, sys
t = json.load(sys.stdin).get(sys.argv[1])
if t is None: sys.exit(f"no instance type {sys.argv[1]}")
print(t["regions_with_capacity_available"][0]["name"] if t["regions_with_capacity_available"] else "")' "$2" <<< "$data")
      [ -n "$region" ] && { echo "$region"; break; }
      sleep "${3:-120}"
    done
    ;;
  add-key)
    pub=$(cat "${2:-$HOME/.ssh/id_ed25519.pub}")
    call POST /ssh-keys "$(python3 -c 'import json, sys; print(json.dumps({"name": sys.argv[1], "public_key": sys.argv[2]}))' "$KEY_NAME" "$pub")" > /dev/null
    echo "registered $KEY_NAME"
    ;;
  launch)
    data=$(call POST /instance-operations/launch "$(python3 -c 'import json, sys; print(json.dumps({"region_name": sys.argv[2], "instance_type_name": sys.argv[1], "ssh_key_names": [sys.argv[3]], "name": "qe-benchmarks-phase2"}))' "$2" "$3" "$KEY_NAME")")
    python3 -c 'import json, sys; print(*json.load(sys.stdin)["instance_ids"])' <<< "$data"
    ;;
  list)
    data=$(call GET /instances)
    python3 -c '
import json, sys
for i in json.load(sys.stdin):
    print(i["id"], i["instance_type"]["name"], i["region"]["name"], i["status"], i.get("ip") or "-")' <<< "$data"
    ;;
  terminate)
    call POST /instance-operations/terminate "$(python3 -c 'import json, sys; print(json.dumps({"instance_ids": [sys.argv[1]]}))' "$2")" > /dev/null
    echo "terminating $2"
    ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
