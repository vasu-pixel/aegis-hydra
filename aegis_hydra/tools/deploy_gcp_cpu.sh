#!/bin/bash
# Deployment script for Aegis Hydra on GCP (CPU Fallback)
# Machine: n2-standard-8 (8 vCPUs, 32GB RAM)
# Zone: us-east4-a (Ashburn, VA) - Low Latency to Coinbase

if [ -z "$1" ]; then
  echo "Usage: ./deploy_gcp_cpu.sh <PROJECT_ID>"
  exit 1
fi

PROJECT_ID=$1
ZONE="us-east4-a"
INSTANCE_NAME="aegis-coinbase-cpu"

echo "Deploying CPU Instance to Project: $PROJECT_ID in Zone: $ZONE"

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=n2-standard-8 \
    --maintenance-policy=MIGRATE \
    --provisioning-model=SPOT \
    --image-family=common-cpu-ubuntu-2204 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB

echo "Instance creation command sent."
echo "Wait for the instance to be ready, then SSH into it:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
