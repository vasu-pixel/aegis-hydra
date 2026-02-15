#!/bin/bash
# Deployment script for Aegis Hydra on GCP (us-east4)

if [ -z "$1" ]; then
  echo "Usage: ./deploy_gcp.sh <PROJECT_ID>"
  exit 1
fi

PROJECT_ID=$1
ZONE="us-east4-a"
INSTANCE_NAME="aegis-coinbase-paper"

echo "Deploying to Project: $PROJECT_ID in Zone: $ZONE"

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --image-family=common-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --metadata="install-nvidia-driver=True"

echo "Instance creation command sent."
echo "Wait for the instance to be ready, then SSH into it:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
