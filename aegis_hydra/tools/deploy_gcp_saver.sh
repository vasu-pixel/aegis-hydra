#!/bin/bash
# Cost-Optimized Deployment (Saver Mode)
# Machine: n1-standard-2 (2 vCPUs, 7.5GB RAM) + T4 GPU
# Zone: us-east4-a (Ashburn, VA)
# Disk: 50GB Standard (HDD)

# Default to known project ID if not provided
PROJECT_ID=${1:-hydra-trading-487421}
ZONE="us-east4-a"
INSTANCE_NAME="aegis-paper-saver"

echo "Deploying Saver Instance to Project: $PROJECT_ID in Zone: $ZONE"

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=n1-standard-2 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --provisioning-model=SPOT \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-standard \
    --metadata="install-nvidia-driver=True"

echo "Instance creation command sent."
echo "Wait for the instance to be ready, then SSH into it:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
