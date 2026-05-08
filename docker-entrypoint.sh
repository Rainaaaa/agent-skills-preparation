#!/usr/bin/env bash
# Dispatch to a pipeline script.
#
#   docker run --rm agent-skills-preparation pipeline/main.py normalize ...
#   docker run --rm agent-skills-preparation pipeline/enrich_t3c.py ...
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- --help
fi

case "$1" in
  --help|-h)
    cat <<'EOF'
agent-skills-preparation container

Pipeline entry points (each accepts --help):

    pipeline/main.py normalize         # canonical → normalized (+ optional scan filter)
    pipeline/main.py build_phase1      # normalized → full_cpt
    pipeline/main.py build_phase2_hcl  # full_cpt   → pl_hcl
    pipeline/enrich_t3c.py             # rule-based T3c enrichment (CPU)

T3a enrichment (vLLM, GPU) is NOT included in this image. Build a GPU
image yourself (see Dockerfile comments) or run T3a outside Docker.

Mount these from the host:
    -v $(pwd)/config.yaml:/app/config.yaml:ro
    -v /path/to/canonical/inputs:/data/inputs:ro
    -v /path/to/prepared:/data/prepared
    -v /path/to/scanning_outputs:/data/scanning:ro
    -e AGENTSKILLS_SCAN_RESULTS=/data/scanning/unified_results.csv
EOF
    exit 0
    ;;
esac

# Same dispatch rule as the sibling repos.
script="$1"; shift
if [[ "$script" == *.py ]]; then
  if [ -f "/app/$script" ]; then
    cd /app
    exec python -u "$script" "$@"
  fi
  if [ -f "$script" ]; then
    exec python -u "$script" "$@"
  fi
  echo "[entrypoint] python script not found: $script" >&2
  exit 64
fi
exec "$script" "$@"
