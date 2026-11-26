#!/bin/bash
Z=${1:-us-east1-d}
echo "create requested $(date -u '+%F %T UTC') ($(TZ=Australia/Sydney date +%H:%M:%S) AEDT) in $Z"
gcloud --configuration=qe-rental compute instances create tpu-f64-stage1 \
  --zone=$Z --machine-type=ct6e-standard-1t \
  --image-project=ubuntu-os-accelerator-images --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --no-service-account --no-scopes \
  --metadata=ssh-keys="ubuntu:$(cat ~/.ssh/id_ed25519.pub)" 2>&1
rc=$?
echo "create returned $rc at $(date -u '+%F %T UTC')"
exit $rc
