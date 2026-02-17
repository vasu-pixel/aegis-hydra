#!/bin/bash
# setup_remote.sh - To be run on the GCP instance
set -e

echo "=== AEGIS HYDRA: REMOTE SETUP ==="

# 1. Update and Install System Dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential python3-pip python3-venv tmux htop

# 2. Setup Python Virtual Environment
echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin_activate
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# 3. Build C++ Daemon
echo "Building C++ components..."
cd aegis_hydra/cpp
make clean
make
cd ../..

echo "✅ Remote setup complete!"
echo "To run the portfolio:"
echo "1. tmux new -s aegis"
echo "2. source .venv/bin/activate"
echo "3. python3 launcher.py"
