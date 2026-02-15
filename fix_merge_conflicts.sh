#!/bin/bash
# Fix merge conflicts and pull latest code

echo "🔧 Fixing merge conflicts..."

# Stash untracked files that are blocking the merge
echo "Moving conflicting files to backup..."
mkdir -p ~/aegis-hydra-backup
mv ~/aegis-hydra/.env ~/aegis-hydra-backup/ 2>/dev/null
mv ~/aegis-hydra/aegis_hydra/cpp/aegis_daemon ~/aegis-hydra-backup/ 2>/dev/null
mv ~/aegis-hydra/aegis_hydra/cpp/libising.so ~/aegis-hydra-backup/ 2>/dev/null
mv ~/aegis-hydra/aegis_hydra/cpp/libising_engine.so ~/aegis-hydra-backup/ 2>/dev/null
mv ~/aegis-hydra/live_dashboard.png ~/aegis-hydra-backup/ 2>/dev/null
mv ~/aegis-hydra/paper_log.csv ~/aegis-hydra-backup/ 2>/dev/null
mv ~/aegis-hydra/paper_state.json ~/aegis-hydra-backup/ 2>/dev/null
mv ~/aegis-hydra/paper_trades.csv ~/aegis-hydra-backup/ 2>/dev/null

echo "✅ Backup created at ~/aegis-hydra-backup/"
echo ""
echo "Now run:"
echo "  cd ~/aegis-hydra"
echo "  git reset --hard HEAD"
echo "  git pull"
echo "  cd aegis_hydra/cpp && make clean && make"
echo "  python3 -m aegis_hydra.tools.hft_pipe"
