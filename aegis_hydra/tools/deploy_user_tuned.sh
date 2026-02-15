#!/bin/bash
# User-Tuned Deployment Script
# Config: n1-standard-4 + T4 + Spot
# Region: us-east4-a

gcloud compute instances create aegis-paper-trader \
    --project=hydra-trading-487421 \
    --zone=us-east4-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --provisioning-model=SPOT \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --metadata="install-nvidia-driver=True"
