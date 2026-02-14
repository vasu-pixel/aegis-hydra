#!/bin/bash
# Deployment script for Aegis Hydra on GCP (High-Performance CPU)
# Machine: c2-standard-16 (16 vCPUs, Compute Optimized)
# Zone: us-east4-a (Ashburn, VA)
# Cost: Higher than N1, but necessary for 10M agents without GPU.

if [ -z "$1" ]; then
  echo "Usage: ./deploy_gcp_c2.sh <PROJECT_ID>"
  exit 1
fi

PROJECT_ID=$1
ZONE="us-east4-a"
INSTANCE_NAME="aegis-hydra-c2"

echo "Deploying Compute-Optimized Instance to Project: $PROJECT_ID in Zone: $ZONE"

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=c2-standard-8 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB

echo "Instance creation command sent."
echo "Wait for the instance to be ready, then SSH into it:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
