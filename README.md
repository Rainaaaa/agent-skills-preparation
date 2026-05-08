# agent-skills-preparation

Versioned data-preparation pipeline for the AgentSkills-OSS misalignment
project. Turns the upstream **canonical_skill_records.jsonl** into the
training-ready dataset versions consumed by the model trainers, with a
**verdict filter** that splits the corpus into two streams:

- **Clean skills** → synthesized into T1/T2/T3 negatives for **model
  training, validation, and testing**.
- **Misaligned / malicious skills** → set aside as the **real-world
  downstream-evaluation corpus** (the audit trail in
  `excluded_skill_ids.txt`).

The verdict source is configurable: it can be the auto-scan output from
`agent-skills-scanning` while human review is in progress, then swapped
for a human-reviewed CSV once review is complete. The pipeline never
hardcodes a path.

### Roles of the synthesized corruption types

| Type | What it is                                | Used for                                |
| ---- | ----------------------------------------- | --------------------------------------- |
| T1   | Internal layer permutation (same skill, wrong order) | Continual pretraining (CPT)        |
| T2   | External layer swap (donor skills)        | Contrastive learning                    |
| T3a  | LLM span-based behavior corruption        | Contrastive learning                    |
| T3b  | Span swap from same-pool donor            | Contrastive learning                    |
| T3c  | Rule-based hallucinated-identifier inj.   | Contrastive learning                    |

Run `python -m pipeline.balance_phase2 ...` after the enrichment stages
to get **equal-distribution** balanced views for each split, so no one
type dominates the trainer's batches. See [Balanced view](#balanced-view)
below.

```
                canonical_skill_records.jsonl
                              │
                              │  [optional] unified_results.csv
                              │              from agent-skills-scanning
                              ▼
                ┌──────────────────────────────┐
   Stage A      │   normalize  (+ scan filter) │  → normalized/<v>/
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
   Stage B      │   build_phase1               │  → full_cpt/<v>/
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
   Stage C2     │   build_phase2_hcl           │  → pl_hcl/<v>/  (T1 / T2 / T3b)
                └──────────────┬───────────────┘
                  ┌────────────┴────────────┐
                  ▼                         ▼
   Stage C3 ┌──────────────┐    Stage C4 ┌──────────────┐
            │  enrich_t3a  │             │  enrich_t3c  │
            │  (LLM, GPU)  │             │  (rule, CPU) │
            └──────────────┘             └──────────────┘
                  │                         │
                  ▼                         ▼
       pl_hcl/<v>/.../*_t3a.parquet    pl_hcl/<v>/.../*_t3c.parquet
```

The two enrichments produce **disjoint negative types** and run
independently. See the per-stage README sections below for the
construction details.

## What's new vs. the legacy `AgentSkills-preparation/data_preparation/`

| Change | Why |
| --- | --- |
| **Verdict filter** in `normalize` | Skills classified as malicious or misaligned (by [agent-skills-scanning](https://github.com/Rainaaaa/agent-skills-scanning) auto-scan **or a human-reviewed CSV with the same schema**) are dropped *before* the dataset is built. Configurable policy under `filter:` in `config.yaml`. The dropped list lands in `excluded_skill_ids.txt` so it can be reused as a real-world evaluation corpus. |
| **Binary alignment vocab** | The alignment axis emits `{ALIGNED, MISALIGNED, ERROR}` instead of `{SAFE, SUSPICIOUS, MALICIOUS, ERROR}`. Aligns with the upstream scanner; severity stays in the verdict's `raw` payload for ablations. |
| **NEW** `pipeline/balance_phase2.py` | Equal-distribution sampler. Emits a `balanced_cpt` view (positives + T1) and a `balanced_contrastive` view (T2 + T3a + T3b + T3c) per split, downsampled to equal counts per type. Reproducible via the same seed. |
| **YAML config + `${VAR:-default}` interpolation** | Same convention as agent-skills-collection / -scanning. Cloners can run with env vars, no need to edit YAML for every install. |
| **Package layout** | Source moves from `src/` to `pipeline/` (a real Python package). `python -m pipeline.main normalize ...` — no more `sys.path` hacks. |
| **Deprecated `build_phase2` (legacy pair-based) and `build_phase3` removed** | Per the previous README's deprecation note: PD-HMCL Phase 2 and the SFT Phase 3 path were already off the live training flow. |
| **Dockerfile + docker-compose** | Cloners get a one-command image build. (CPU-only image — T3a's vLLM stack is opt-in via a separate GPU image.) |

## Layout

```
agent-skills-preparation/
├── README.md                  # this file
├── Dockerfile                 # CPU image: normalize + phase1 + phase2_hcl + t3c
├── docker-compose.yml         # named services per stage
├── docker-entrypoint.sh
├── .dockerignore
├── .gitignore
├── requirements.txt
├── config.example.yaml        # template (gitignored: config.yaml)
├── cleanup_local.sh
│
├── pipeline/                  # the Python package
│   ├── _shared.py             # IO + config (YAML + env interp) + manifest helpers
│   ├── scan_filter.py         # NEW: drop malicious/misaligned skills
│   ├── schema.py              # canonical → normalized + sample builders
│   ├── _corrupt_spans.py      # span tokenize/score/sample/splice (T3a/T3c shared)
│   ├── normalize.py           # Stage A — embeds the scan filter
│   ├── build_phase1.py        # Stage B
│   ├── build_phase2_hcl.py    # Stage C2 — T1 / T2 / T3b deterministic
│   ├── enrich_t3a.py          # Stage C3 — LLM span-based behavior corruption
│   ├── enrich_t3c.py          # Stage C4 — rule-based hallucinated identifiers
│   └── main.py                # CLI dispatcher (subcommands: normalize, build_phase1, build_phase2_hcl)
│
├── scripts/                   # SLURM wrappers + smoke probes
│   ├── run_data_preparation.sh
│   ├── run_enrich_t3a.sh
│   ├── run_enrich_t3c.sh
│   ├── submit_all.sh
│   ├── cuda_driver_probe.sh
│   └── smoke_test_vllm.py
│
├── inputs/                    # gitignored
└── outputs/                   # gitignored (logs/ + manifests/)
```

## First-time setup

```bash
# 1. Make your local config
cp config.example.yaml config.yaml

# 2. Either edit config.yaml directly OR set env vars (recommended):
export AGENTSKILLS_CANONICAL_RECORDS=/path/to/canonical_skill_records.jsonl
export AGENTSKILLS_PREPARED_ROOT=/path/to/datasets/misalignment
export AGENTSKILLS_SCAN_RESULTS=/path/to/agent-skills-scanning/outputs/unified_results.csv

# 3. Local Python install (pyarrow is the heaviest dep; T3a vLLM is opt-in)
pip install -r requirements.txt
```

## Running

### Stage A — Normalize (with verdict filter)

```bash
python -m pipeline.main normalize \
    --config config.yaml \
    --normalized-version v1 \
    --scan-results /path/to/verdicts.csv     # optional; falls back to env / config
```

`--scan-results` is the single configuration knob. Point it at:

- The **auto-scan** output from `agent-skills-scanning`
  (`outputs/unified_results.csv`) while human review is still in progress.
- A **human-reviewed CSV** once that's ready. Same schema:

  ```csv
  skill_id,overall_class,alignment_class
  owner-repo-skill-md,MALICIOUS,MISALIGNED
  another-skill,SAFE,ALIGNED
  ```

The filter logs how many skills were dropped per axis:

```
[scan_filter] policy: drop_overall=['MALICIOUS', 'SUSPICIOUS']
              drop_alignment=['MISALIGNED']
              unscanned_action=keep
[scan_filter] excluded_by_overall=2143  excluded_by_alignment=178
              excluded_by_either=2204  kept=263840
              unscanned_kept=14  unscanned_dropped=0
```

The drop counts land in `normalized/<v>/stats.json > scan_filter.report`,
and the **list of dropped skill_ids** is written to
`normalized/<v>/excluded_skill_ids.txt`. That file is the seed for the
real-world downstream-evaluation corpus — the model is judged on exactly
those skills (refined through human review later if needed).

#### Filter policy

Default: drop a skill if `overall_class ∈ {MALICIOUS, SUSPICIOUS}` OR
`alignment_class == MISALIGNED`.

Override in `config.yaml`:

```yaml
filter:
  scan_results: "${AGENTSKILLS_SCAN_RESULTS:-}"
  exclude_overall_classes:   ["MALICIOUS"]    # keep SUSPICIOUS-only skills
  exclude_alignment_classes: ["MISALIGNED"]
  unscanned_action: "drop"                    # strict — every skill must have a verdict
```

If `scan_results` is empty / unset / file missing, the filter is a
**no-op** and normalize processes the full canonical set (handy for a
first pass before any verdicts exist).

### Stage B — Phase 1 (Structured CLM CPT)

```bash
python -m pipeline.main build_phase1 \
    --config config.yaml \
    --normalized-version v1 \
    --output-version full_cpt_v1
```

### Stage C2 — Phase 2 HCL

```bash
python -m pipeline.main build_phase2_hcl \
    --config config.yaml \
    --full-cpt-version full_cpt_v1 \
    --output-version pl_hcl_v1
```

### Stage C3 / C4 — Enrichments

```bash
# T3a — LLM span-based behavior corruption (vLLM, GPU)
python -m pipeline.enrich_t3a \
    --pl-hcl-root /path/to/datasets/misalignment/pl_hcl \
    --output-version pl_hcl_v1 \
    --model-path /path/to/llama3.1-8b-Instruct/snapshot \
    --tensor-parallel-size 1 --k-spans 2 --span-lines 8 \
    --splits train,val,test

# T3c — rule-based hallucinated-identifier injection (CPU only)
python -m pipeline.enrich_t3c \
    --pl-hcl-root /path/to/datasets/misalignment/pl_hcl \
    --output-version pl_hcl_v1 \
    --k-subs 2 --splits train,val,test
```

T3a and T3c read the **same** Phase 2 HCL positives but write to
**different** files (`*_t3a.parquet` vs `*_t3c.parquet`), so they can
run in parallel.

### Balanced view

After the enrichment stages finish, every type has a different per-split
row count. Run the balanced sampler to materialize equal-distribution
training views:

```bash
python -m pipeline.balance_phase2 \
    --pl-hcl-root  /path/to/datasets/misalignment/pl_hcl \
    --full-cpt-root /path/to/datasets/misalignment/full_cpt \
    --pl-hcl-version  pl_hcl_v1 \
    --full-cpt-version full_cpt_v1 \
    --splits train,val,test
```

Output:

| Path | What it contains |
| ---- | ---------------- |
| `full_cpt/<v>/<stage>/<split>_balanced_cpt.parquet`            | positives + T1, equal counts per type — for **continual pretraining** |
| `pl_hcl/<v>/<stage>/<split>_balanced_contrastive.parquet`      | T2 + T3a + T3b + T3c, equal counts per type — for **contrastive learning** |
| `pl_hcl/<v>/stats_balanced.json`                                | Per-(stage, split) sampling summary (kept counts, target, seed) |

By default each type is downsampled to `min(rows per type)` for that
split, so the balanced view's row count is `n_types × min_count`. Set
`--target-rows-per-type N` to cap each type at `N` rows instead.

The sampler is **non-destructive** — the original per-label parquets
(`<split>.parquet`, `<split>_t1.parquet`, …) stay in place for ablations
or per-class weighting.

### Docker

```bash
docker build -t agent-skills-preparation .
docker compose run --rm normalize           # uses default config + env vars
docker compose run --rm build-phase1
docker compose run --rm build-phase2-hcl
docker compose run --rm enrich-t3c
```

T3a is **not** in this image (it pulls vLLM + a GPU PyTorch wheel that
the typical user doesn't need). Build a GPU image yourself by
uncommenting the `vllm` / `torch` lines in `requirements.txt` and using
a CUDA base image, OR run T3a outside Docker.

### SLURM (IU BR200)

[`scripts/run_data_preparation.sh`](scripts/run_data_preparation.sh) is
the SLURM wrapper for `pipeline.main`. Set `PIPELINE_COMMAND` to one of
`normalize` / `build_phase1` / `build_phase2_hcl` plus the matching
`*_VERSION` env vars.

[`scripts/run_enrich_t3a.sh`](scripts/run_enrich_t3a.sh) and
[`scripts/run_enrich_t3c.sh`](scripts/run_enrich_t3c.sh) are dedicated
launchers for the two enrichment scripts.

## Versioning rules

The same rules from the legacy pipeline still apply:

1. `normalize` creates a named normalized version (`v1`, `v2`, …).
2. Each downstream phase explicitly references one normalized version.
3. Each downstream phase writes its own named output version.
4. No phase silently overwrites — pick a new `--output-version` if the
   target already exists.

A `manifest.json` is written into every version dir; a copy lands in
`outputs/manifests/<dataset>__<version>.json` for browsing.

## Pipeline guarantees

- **Reproducible** — every dataset version carries the upstream input
  path, the config summary, file SHA-256s, and (for normalized) the
  scan-filter policy + report.
- **Append-only outputs** — version dirs are created exclusively; pick
  a new version name to re-build with new code or new inputs.
- **Loose coupling** — stages communicate via documented file paths
  under `prepared_root/<dataset>/<version>/`, so you can swap any one
  stage's implementation without touching the others.
