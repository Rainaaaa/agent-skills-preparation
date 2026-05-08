# syntax=docker/dockerfile:1.7
#
# agent-skills-preparation — pipeline image for normalize + build_phase1 +
# build_phase2_hcl + enrich_t3c (CPU-only stages).
#
# **The T3a enrichment is NOT packaged in this image** because it depends
# on vLLM + a GPU-capable PyTorch wheel that the typical pipeline user
# doesn't need. If you want a GPU image, uncomment the vllm/torch lines
# in `requirements.txt` and build with a CUDA base instead of `slim`.
#
# Build:
#     docker build -t agent-skills-preparation .
#
# Run (mount your data + config):
#     docker run --rm \
#         -v $(pwd)/config.yaml:/app/config.yaml:ro \
#         -v /path/to/canonical/inputs:/data/inputs:ro \
#         -v /path/to/prepared:/data/prepared \
#         -v /path/to/scanning_outputs:/data/scanning:ro \
#         -e AGENTSKILLS_SCAN_RESULTS=/data/scanning/unified_results.csv \
#         agent-skills-preparation \
#         pipeline/main.py normalize --config /app/config.yaml --normalized-version v1

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Pipeline source.
COPY pipeline /app/pipeline
COPY scripts  /app/scripts
COPY config.example.yaml /app/config.example.yaml
COPY README.md /app/README.md

# Default config inside the image is the template; users override via
# bind-mounting their own config.yaml or by setting the env vars
# documented in config.example.yaml.
RUN cp /app/config.example.yaml /app/config.yaml

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]
