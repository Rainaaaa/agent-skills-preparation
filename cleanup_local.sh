#!/usr/bin/env bash
# Optional local cleanup — wipes prepared artifacts + Python caches.
#
#   bash cleanup_local.sh                # actually delete
#   DRYRUN=1 bash cleanup_local.sh       # preview only

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
  if [ "${DRYRUN:-0}" = "1" ]; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    "$@"
  fi
}

cd "$ROOT"

# Local outputs (manifests, logs, smoke runs)
[ -d outputs ] && run rm -rf outputs && run mkdir -p outputs && run touch outputs/.gitkeep

# Python + IDE caches
find . -name __pycache__   -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

# Backups + tmp
find . -name '*.bak' -delete 2>/dev/null || true
find . -name '*.tmp' -delete 2>/dev/null || true

# Local secrets
[ -f .env ] && run rm -f .env

# The pre-refactor AgentSkills-preparation/ tree has been replaced —
# delete it manually when you're sure the new pipeline works:
PARENT="$(dirname "$ROOT")"
if [ -d "$PARENT/AgentSkills-preparation" ]; then
  cat <<EOF

NOTE: The pre-refactor AgentSkills-preparation/ tree still exists at:
    $PARENT/AgentSkills-preparation
This new agent-skills-preparation/ replaces it. Delete with:
    rm -rf $PARENT/AgentSkills-preparation
(skipped here so you can review first.)
EOF
fi

echo "[done] cleanup complete."
