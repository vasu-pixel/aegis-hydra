#!/bin/bash
# Run HFT pipeline with CPU pinning for minimal latency

echo "=== AEGIS-HYDRA HFT LAUNCHER (CPU PINNED) ==="

# Check if taskset is available
if ! command -v taskset &> /dev/null; then
    echo "❌ taskset not found. Install with: sudo apt-get install util-linux"
    exit 1
fi

# Get number of CPUs
NUM_CPUS=$(nproc)
echo "Available CPUs: $NUM_CPUS"

# Pin to CPUs 0-3 (physical cores, avoid hyperthreading)
# Adjust if you have fewer cores
if [ $NUM_CPUS -ge 4 ]; then
    CPU_MASK="0-3"
    echo "Pinning to CPUs: $CPU_MASK"
else
    CPU_MASK="0-$((NUM_CPUS-1))"
    echo "Pinning to CPUs: $CPU_MASK"
fi

# Set real-time priority (optional, requires permissions)
# PRIORITY="chrt -r 50"  # Uncomment if you have RT permissions
PRIORITY=""

# Run with CPU pinning
echo "Starting HFT pipeline..."
echo "Command: taskset -c $CPU_MASK $PRIORITY python3 -m aegis_hydra.tools.hft_pipe"
echo ""

cd ~/aegis-hydra
exec taskset -c $CPU_MASK $PRIORITY python3 -m aegis_hydra.tools.hft_pipe
