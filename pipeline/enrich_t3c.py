"""Phase 2 HCL — Step C: T3c rule-based hallucinated-identifier injection.

For each anchor (label='positive') in
``pl_hcl/<output_version>/stage{1,2}/{train,val,test}.parquet`` (skips
``unseen`` — donor-only pool), apply K deterministic identifier
substitutions to the target layer (INSTRUCTION for stage1; INSTRUCTION
+ RESOURCE for stage2). Substitutions replace real package / library /
file / CLI / env identifiers with plausible-looking but non-existent
variants — Example A in the slides ('@nx/react-codeshift' is a phantom
package) is the canonical pattern.

Why rule-based, not LLM
-----------------------
* **Speed**: CPU-only, finishes in minutes. No GPU needed.
* **Ground truth**: by construction, the new identifier does not exist
  in the relevant package registry. The dataset paper can document the
  registry version and reproduce results exactly.
* **Disjoint from T3a**: T3a edits behavior on real identifiers (still
  importable, runtime divergent). T3c edits identifiers and keeps
  behavior intent (importable → fail; runtime never reached). The two
  failure modes are orthogonal.

Output
------
``pl_hcl/<output_version>/stage{1,2}/{train,val,test}_t3c.parquet``,
with rows matching the rest of the pl_hcl schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline._corrupt_spans import compose_layers, est_tokens, extract_layer, wrap

LOGGER = logging.getLogger("enrich_phase2_t3c")


# ---------------------------------------------------------------------------
# Substitution registry
#
# Each entry maps a frequently-seen REAL identifier to a plausible-but-fake
# variant. When the same identifier appears in multiple ecosystems the regex
# context (group 0 / group 1) decides which variant applies.
#
# Versioned: bump REGISTRY_VERSION whenever entries change so the dataset
# paper can cite a stable snapshot.
# ---------------------------------------------------------------------------

REGISTRY_VERSION = 2

# Curated (real → fake) substitutions. Keep ALL keys lowercase for matching.
PIP_SUBS: Dict[str, str] = {
    "pandas": "pandasx",
    "numpy": "numpyz",
    "requests": "requestz",
    "scipy": "scipyx",
    "scikit-learn": "scikit-learn-x",
    "sklearn": "sklearnx",
    "torch": "torchx",
    "tensorflow": "tensorflox",
    "transformers": "transformerx",
    "pyarrow": "pyarrowx",
    "matplotlib": "matplotlibx",
    "beautifulsoup4": "beautifulsoup5",
    "lxml": "lxml-pro",
    "pillow": "pillowx",
    "openpyxl": "openpyxl-x",
    "click": "clickx",
    "tqdm": "tqdmx",
    "pyyaml": "pyyaml-pro",
    "fastapi": "fastapix",
    "uvicorn": "uvicornx",
    "pydantic": "pydanticx",
    "openai": "openaix",
    "anthropic": "anthropicx",
    "boto3": "boto4",
    "redis": "redisx",
    "psycopg2": "psycopg3-fork",
    "sqlalchemy": "sqlalchemyx",
}

NPM_SUBS: Dict[str, str] = {
    "lodash": "lodash-extra",
    "chalk": "chalkx",
    "axios": "axios-pro",
    "react": "react-fast",
    "react-dom": "react-dom-fast",
    "vue": "vuex-core",
    "express": "expressx",
    "next": "nextjs-pro",
    "webpack": "webpack5-fork",
    "typescript": "typescript-pro",
    "eslint": "eslint-extra",
    "prettier": "prettierx-fmt",
    "@types/node": "@types/node-pro",
    "@nx/react": "@nx/react-codeshift",
    "@nx/devkit": "@nx/devkit-pro",
    "@angular/core": "@angular/core-pro",
    "uuid": "uuid-pro",
}

CLI_FLAG_SUBS: Dict[str, str] = {
    "--force": "--force-strict",
    "--force-with-lease": "--safety-lease",
    "--quiet": "--ultra-quiet",
    "--verbose": "--ultra-verbose",
    "--dry-run": "--dry-strict",
    "--no-cache": "--zero-cache",
    "--recursive": "--deep-recursive",
}

ENV_VAR_SUBS: Dict[str, str] = {
    "OPENAI_API_KEY": "OPENAI_SECRET_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_SECRET_KEY",
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_TOKEN",
    "AWS_SECRET_ACCESS_KEY": "AWS_PRIVATE_TOKEN",
    "GITHUB_TOKEN": "GITHUB_ACCESS_TOKEN",
    "HF_TOKEN": "HF_API_KEY",
    "DATABASE_URL": "DB_CONNECTION_URL",
    "REDIS_URL": "REDIS_CONNECTION",
}


# Generic plausible suffixes used when an identifier isn't in the curated
# registry but matches a structural pattern (e.g. obviously a python
# package name). Picked deterministically per (kind, name) so each
# unique identifier always corrupts to the same fake (reproducibility).
_GENERIC_SUFFIXES = ["-x", "-pro", "z", "-plus", "-ng", "-lite", "2"]


def _generic_fake(name: str, kind: str = "") -> str:
    """Append a deterministic plausible suffix to a name not in the registry."""
    h = hashlib.sha1(f"{kind}|{name}".encode("utf-8")).digest()
    suffix = _GENERIC_SUFFIXES[h[0] % len(_GENERIC_SUFFIXES)]
    return f"{name}{suffix}"


# Python stdlib modules that look like packages but must NEVER be corrupted —
# importing a fake stdlib name is a different category of misalignment and
# would produce many spurious "obviously broken" samples.
_PY_STDLIB = frozenset({
    "os", "sys", "re", "json", "math", "time", "datetime", "logging",
    "argparse", "pathlib", "hashlib", "random", "collections", "typing",
    "subprocess", "shutil", "io", "csv", "uuid", "tempfile", "functools",
    "itertools", "importlib", "warnings", "traceback", "string", "copy",
    "pickle", "base64", "urllib", "asyncio", "threading", "multiprocessing",
    "queue", "socket", "struct", "enum", "abc", "contextlib", "dataclasses",
    "ast", "inspect", "types", "operator", "weakref", "glob", "platform",
    "signal", "select", "errno", "stat", "fcntl", "gc", "atexit", "uuid",
    "secrets", "hmac", "zlib", "gzip", "bz2", "lzma", "zipfile", "tarfile",
    "configparser", "decimal", "fractions", "statistics", "textwrap",
    "unicodedata", "html", "xml", "http", "email",
})

# A "name looks plausibly like a package" check: lowercase letters, digits,
# - and _ only, length ≥ 3, doesn't start with a digit. Filters out single
# letters, weird punctuation, etc.
_PKG_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{2,}$")
def _looks_like_pkg(name: str) -> bool:
    return bool(_PKG_NAME_RE.match(name)) and name not in _PY_STDLIB


# ---------------------------------------------------------------------------
# Substitution rules — each rule is (regex, finder, applier).
#
# A rule scans the entire layer text and returns a list of `Hit` candidates.
# Each Hit is a (start_char, end_char, replacement_text, kind) tuple.
# The driver picks K hits deterministically and splices replacements.
# ---------------------------------------------------------------------------

_PY_IMPORT = re.compile(r"^(\s*)(import|from)\s+([\w.]+)", re.MULTILINE)
_NPM_DEP_LINE = re.compile(r'"((?:@[\w-]+/)?[\w.-]+)"\s*:\s*"[^"]+"')
_PIP_PIN = re.compile(r"^([\w.-]+)\s*(==|>=|<=|~=)\s*([\w.\-+]+)", re.MULTILINE)
_REQUIRE_CALL = re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
_NPM_INSTALL_CMD = re.compile(r"\b(npm|yarn|pnpm)\s+(install|add)\s+([@\w./-]+)")
_PIP_INSTALL_CMD = re.compile(r"\bpip\s+install\s+([\w.\-+]+)")
_CLI_FLAG = re.compile(r"--[a-z][\w-]+")
_ENV_VAR = re.compile(r"\b([A-Z][A-Z0-9_]{4,})\b")


def _find_pip_subs(text: str) -> List[Tuple[int, int, str, str]]:
    """Find Python `import X` / `from X import` lines targeting registry pkgs.

    Falls back to a deterministic generic fake for non-stdlib packages
    that aren't in PIP_SUBS (kind='py_import_generic').
    """
    hits = []
    for m in _PY_IMPORT.finditer(text):
        indent, kw, mod = m.group(1), m.group(2), m.group(3)
        top = mod.split(".")[0].lower()
        if top in PIP_SUBS:
            new_top, kind = PIP_SUBS[top], "py_import"
        elif _looks_like_pkg(top):
            new_top, kind = _generic_fake(top, "py_import"), "py_import_generic"
        else:
            continue
        new_mod = new_top + mod[len(top):]
        new_line = f"{indent}{kw} {new_mod}"
        hits.append((m.start(), m.end(), new_line, kind))
    return hits


def _find_pip_pin_subs(text: str) -> List[Tuple[int, int, str, str]]:
    hits = []
    for m in _PIP_PIN.finditer(text):
        name, op, ver = m.group(1), m.group(2), m.group(3)
        key = name.lower()
        if key in PIP_SUBS:
            new_name, kind = PIP_SUBS[key], "pip_pin"
        elif _looks_like_pkg(key):
            new_name, kind = _generic_fake(key, "pip_pin"), "pip_pin_generic"
        else:
            continue
        new_line = f"{new_name}{op}{ver}"
        hits.append((m.start(), m.end(), new_line, kind))
    return hits


def _find_npm_dep_subs(text: str) -> List[Tuple[int, int, str, str]]:
    hits = []
    for m in _NPM_DEP_LINE.finditer(text):
        name = m.group(1)
        key = name.lower()
        if key in NPM_SUBS:
            new_name, kind = NPM_SUBS[key], "npm_dep"
        elif _looks_like_pkg(key.lstrip("@").split("/")[-1] if key.startswith("@") else key):
            new_name, kind = _generic_fake(name, "npm_dep"), "npm_dep_generic"
        else:
            continue
        new = m.group(0).replace(f'"{name}"', f'"{new_name}"', 1)
        hits.append((m.start(), m.end(), new, kind))
    return hits


def _find_require_subs(text: str) -> List[Tuple[int, int, str, str]]:
    hits = []
    for m in _REQUIRE_CALL.finditer(text):
        name = m.group(1)
        key = name.lower()
        if key in NPM_SUBS:
            new_name, kind = NPM_SUBS[key], "js_require"
        elif _looks_like_pkg(key.lstrip("@").split("/")[-1] if key.startswith("@") else key):
            new_name, kind = _generic_fake(name, "js_require"), "js_require_generic"
        else:
            continue
        new = m.group(0).replace(name, new_name, 1)
        hits.append((m.start(), m.end(), new, kind))
    return hits


def _find_install_cmd_subs(text: str) -> List[Tuple[int, int, str, str]]:
    hits = []
    for m in _NPM_INSTALL_CMD.finditer(text):
        mgr, sub, pkg = m.group(1), m.group(2), m.group(3)
        key = pkg.lower()
        if key in NPM_SUBS:
            new_name, kind = NPM_SUBS[key], "npm_cmd"
        elif _looks_like_pkg(key.lstrip("@").split("/")[-1] if key.startswith("@") else key):
            new_name, kind = _generic_fake(pkg, "npm_cmd"), "npm_cmd_generic"
        else:
            continue
        new = f"{mgr} {sub} {new_name}"
        hits.append((m.start(), m.end(), new, kind))
    for m in _PIP_INSTALL_CMD.finditer(text):
        pkg = m.group(1)
        key = pkg.lower()
        if key in PIP_SUBS:
            new_name, kind = PIP_SUBS[key], "pip_cmd"
        elif _looks_like_pkg(key):
            new_name, kind = _generic_fake(key, "pip_cmd"), "pip_cmd_generic"
        else:
            continue
        new = f"pip install {new_name}"
        hits.append((m.start(), m.end(), new, kind))
    return hits


def _find_cli_flag_subs(text: str) -> List[Tuple[int, int, str, str]]:
    """Match `--xxx` flags. Curated → CLI_FLAG_SUBS; otherwise generic suffix."""
    hits = []
    for m in _CLI_FLAG.finditer(text):
        flag = m.group(0)
        if flag in CLI_FLAG_SUBS:
            hits.append((m.start(), m.end(), CLI_FLAG_SUBS[flag], "cli_flag"))
        else:
            hits.append((m.start(), m.end(),
                         _generic_fake(flag, "cli_flag"), "cli_flag_generic"))
    return hits


def _find_env_var_subs(text: str) -> List[Tuple[int, int, str, str]]:
    hits = []
    for m in _ENV_VAR.finditer(text):
        v = m.group(1)
        if v in ENV_VAR_SUBS:
            hits.append((m.start(), m.end(), ENV_VAR_SUBS[v], "env_var"))
    return hits


_FINDERS = (
    _find_pip_subs,
    _find_pip_pin_subs,
    _find_npm_dep_subs,
    _find_require_subs,
    _find_install_cmd_subs,
    _find_cli_flag_subs,
    _find_env_var_subs,
)


def _enumerate_hits(text: str) -> List[Tuple[int, int, str, str]]:
    hits: List[Tuple[int, int, str, str]] = []
    for f in _FINDERS:
        hits.extend(f(text))
    # Sort by start; drop overlapping (keep first).
    hits.sort(key=lambda h: (h[0], h[1]))
    deduped: List[Tuple[int, int, str, str]] = []
    last_end = -1
    for h in hits:
        if h[0] >= last_end:
            deduped.append(h)
            last_end = h[1]
    return deduped


def _seeded_rng(*parts: str) -> random.Random:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:8], "big", signed=False))


def _apply_subs(
    text: str,
    hits: List[Tuple[int, int, str, str]],
    k: int,
    rng: random.Random,
) -> Tuple[str, List[str]]:
    """Apply up to K hits (chosen randomly with the seeded RNG)."""
    if not hits or k <= 0:
        return text, []
    chosen_idx = sorted(rng.sample(range(len(hits)), k=min(k, len(hits))))
    out = []
    cursor = 0
    kinds: List[str] = []
    for i in chosen_idx:
        s, e, repl, kind = hits[i]
        if s < cursor:
            continue  # safety
        out.append(text[cursor:s])
        out.append(repl)
        kinds.append(kind)
        cursor = e
    out.append(text[cursor:])
    return "".join(out), kinds


# ---------------------------------------------------------------------------
# Discovery / IO — mirrors enrich_phase2_t3a.py
# ---------------------------------------------------------------------------

STAGE_T3C_LAYERS: Dict[str, List[str]] = {
    "stage1": ["I"],
    "stage2": ["I", "R"],
}
STAGE_MAX_TOKENS: Dict[str, int] = {"stage1": 4096, "stage2": 10240}
SPLITS = ("train", "val", "test")


def _read_positives(parquet_path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore
    cols = [
        "anchor_skill_id", "label", "pool", "split", "stage", "text",
        "normalized_version", "phase1_output_version", "output_version",
    ]
    sch_cols = pq.read_schema(parquet_path).names
    cols = [c for c in cols if c in sch_cols]
    t = pq.read_table(parquet_path, columns=cols)
    return [r for r in t.to_pylist() if r.get("label") == "positive"]


def _existing_keys(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    try:
        import pyarrow.parquet as pq  # type: ignore
        t = pq.read_table(out_path, columns=["anchor_skill_id", "target_layer"])
        return set(zip(t.column("anchor_skill_id").to_pylist(),
                       t.column("target_layer").to_pylist()))
    except Exception as e:
        LOGGER.warning("could not read %s: %s", out_path, e)
        return set()


def _compose_with_swap(anchor: Dict[str, Any], layer: str, new_inner: str, stage: str) -> str:
    full = anchor.get("text", "") or ""
    order = ("M", "I") if stage == "stage1" else ("M", "I", "R")
    parts = {ll: extract_layer(full, ll) for ll in order}
    parts[layer] = new_inner
    return compose_layers(parts, stage)


def _make_row(
    *,
    anchor: Dict[str, Any],
    layer: str,
    kinds: List[str],
    text: str,
    output_version: str,
) -> Dict[str, Any]:
    sub = (
        f"layer={layer}|method=rule|registry_v={REGISTRY_VERSION}|"
        f"k={len(kinds)}|kinds={','.join(kinds)}"
    )
    sample_id = hashlib.sha1(
        f"T3c|{sub}|{anchor['anchor_skill_id']}|{text[:128]}".encode("utf-8")
    ).hexdigest()
    return {
        "sample_id": sample_id,
        "anchor_skill_id": anchor["anchor_skill_id"],
        "label": "T3c",
        "sub_strategy": sub,
        "target_layer": layer,
        "source_skill_ids": [anchor["anchor_skill_id"]],
        "text": text,
        "text_chars": len(text),
        "text_tokens_est": est_tokens(text),
        "pool": anchor.get("pool", ""),
        "split": anchor.get("split", ""),
        "stage": anchor.get("stage", ""),
        "normalized_version": anchor.get("normalized_version", ""),
        "phase1_output_version": anchor.get("phase1_output_version", ""),
        "output_version": output_version,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    pl_hcl_root: Path,
    output_version: str,
    k_subs: int,
    splits: List[str],
    force: bool = False,
) -> None:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    target_root = pl_hcl_root / output_version
    if not target_root.exists():
        raise FileNotFoundError(f"pl_hcl version not found: {target_root}")

    summary: Dict[str, Any] = {
        "method": "rule-based identifier substitution",
        "registry_version": REGISTRY_VERSION,
        "k_subs": k_subs,
        "splits": splits,
        "started_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "per_stage_split": {},
    }

    rng_label = f"T3c|{output_version}|registry_v={REGISTRY_VERSION}|k={k_subs}"

    for stage in ("stage1", "stage2"):
        stage_dir = target_root / stage
        if not stage_dir.exists():
            continue
        max_total = STAGE_MAX_TOKENS[stage]
        layers = STAGE_T3C_LAYERS[stage]

        for split in splits:
            in_path = stage_dir / f"{split}.parquet"
            out_path = stage_dir / f"{split}_t3c.parquet"
            if not in_path.exists():
                continue
            anchors = _read_positives(in_path)
            already = set() if force else _existing_keys(out_path)
            LOGGER.info("[%s/%s] anchors=%d already=%d (force=%s)",
                        stage, split, len(anchors), len(already), force)

            buffer: List[Dict[str, Any]] = []
            writer: Optional[Any] = None
            FLUSH_EVERY = 2048
            n_emitted = n_no_hits = n_oversize = 0

            def _flush(force: bool = False) -> None:
                nonlocal writer, buffer
                if not buffer or (not force and len(buffer) < FLUSH_EVERY):
                    return
                table = pa.Table.from_pylist(buffer)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
                writer.write_table(table)
                buffer = []

            n_empty_layer = 0
            for a in anchors:
                sid = a["anchor_skill_id"]
                full_anchor_text = a.get("text", "") or ""
                for layer in layers:
                    if (sid, layer) in already:
                        continue
                    inner_text = extract_layer(full_anchor_text, layer)
                    if not inner_text or not inner_text.strip():
                        n_empty_layer += 1
                        continue
                    hits = _enumerate_hits(inner_text)
                    if not hits:
                        n_no_hits += 1
                        continue
                    rng = _seeded_rng(rng_label, sid, layer)
                    new_inner, kinds = _apply_subs(inner_text, hits, k_subs, rng)
                    if not kinds:
                        n_no_hits += 1
                        continue
                    full_text = _compose_with_swap(a, layer, new_inner, stage)
                    if est_tokens(full_text) > max_total:
                        n_oversize += 1
                        continue
                    buffer.append(_make_row(
                        anchor=a, layer=layer, kinds=kinds,
                        text=full_text, output_version=output_version,
                    ))
                    n_emitted += 1
                _flush()

            _flush(force=True)
            if writer is not None:
                writer.close()
            LOGGER.info(
                "[%s/%s] wrote %s — emitted=%d no_hits=%d oversize=%d empty_layer=%d",
                stage, split, out_path, n_emitted, n_no_hits, n_oversize, n_empty_layer,
            )
            summary["per_stage_split"][f"{stage}/{split}"] = {
                "anchors": len(anchors),
                "emitted": n_emitted,
                "no_hits_skipped": n_no_hits,
                "oversize_dropped": n_oversize,
                "empty_layer_skipped": n_empty_layer,
            }

    summary["finished_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary_path = target_root / "stats_t3c.json"
    with summary_path.open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote summary: %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="T3c rule-based identifier hallucination")
    parser.add_argument("--pl-hcl-root", required=True)
    parser.add_argument("--output-version", required=True)
    parser.add_argument("--k-subs", type=int, default=2,
                        help="Number of identifier substitutions per (anchor, layer)")
    parser.add_argument("--splits", default="train,val,test",
                        help="Comma list. 'unseen' is excluded by default.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing T3c parquets — full rebuild "
                             "(use after registry version change).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    run(
        pl_hcl_root=Path(args.pl_hcl_root),
        output_version=args.output_version,
        k_subs=args.k_subs,
        splits=splits,
        force=args.force,
    )


if __name__ == "__main__":
    main()
