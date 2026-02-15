#!/bin/bash
# Rebuild daemon and restart HFT with new thresholds

echo "=== REBUILDING AEGIS DAEMON WITH NEW THRESHOLDS ==="

# Stop any running HFT processes
echo "Stopping existing HFT processes..."
pkill -f "aegis_hydra.tools.hft_pipe" || true
pkill -f "aegis_daemon" || true
sleep 2

# Rebuild C++ daemon
echo "Rebuilding C++ daemon..."
cd ~/aegis-hydra/aegis_hydra/cpp
make clean
make

# Verify the binary was updated
if [ -f "aegis_daemon" ]; then
    echo "✅ New daemon binary created: $(ls -lh aegis_daemon)"
else
    echo "❌ Daemon binary not found!"
    exit 1
fi

# Return to project root and run
cd ~/aegis-hydra
echo ""
echo "=== STARTING ULTRA-HFT MODE ==="
echo "Expected thresholds: ~0.35-0.40 (was: ~0.60-0.80)"
echo ""
python3 -m aegis_hydra.tools.hft_pipe
