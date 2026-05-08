"""Shared utilities for span-based negative construction (T3a / T3c).

The whole-layer rewrite approach is too expensive at scale (~70 GPU-h on a
single A100 for stage2). Span-based corruption keeps the per-anchor signal
density (K spans of 5–12 lines per layer) while cutting input + output
tokens by ~5–8x.

Two scorers select the spans that get fed to each negative type — this is
how the T3a / T3c boundary is enforced:

* `score_t3a(line)` — logic-heavy lines (conditionals, comparisons,
  numeric constants, function calls with args). The model is asked to
  flip behavior on real identifiers there.

* `score_t3c(line)` — identifier-heavy lines (imports, dep entries, CLI
  package names, file paths). The model (or a deterministic rule) is
  asked to fabricate non-existent identifiers there.

Different input content → naturally disjoint outputs without iterating
the prompts.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Layer text helpers (mirror enrich_phase2_t3a.py wrappers)
# ---------------------------------------------------------------------------

_M_INNER = re.compile(r"^<METADATA>\n(.*)\n</METADATA>$", re.DOTALL)
_I_INNER = re.compile(r"^<INSTRUCTION>\n(.*)\n</INSTRUCTION>$", re.DOTALL)
_R_INNER = re.compile(r"^<RESOURCE>\n(.*)\n</RESOURCE>$", re.DOTALL)
_RGX = {"M": _M_INNER, "I": _I_INNER, "R": _R_INNER}

# Non-anchored variants — find a layer block anywhere inside a composed `text`.
_M_BLOCK = re.compile(r"<METADATA>\n(.*?)\n</METADATA>", re.DOTALL)
_I_BLOCK = re.compile(r"<INSTRUCTION>\n(.*?)\n</INSTRUCTION>", re.DOTALL)
_R_BLOCK = re.compile(r"<RESOURCE>\n(.*?)\n</RESOURCE>", re.DOTALL)
_RGX_BLOCK = {"M": _M_BLOCK, "I": _I_BLOCK, "R": _R_BLOCK}


def inner(layer_text: str, layer: str) -> str:
    if not layer_text:
        return ""
    m = _RGX[layer].match(layer_text)
    return m.group(1) if m else layer_text


def extract_layer(full_text: str, layer: str) -> str:
    """Pull the inner text of <LAYER>...</LAYER> out of a composed row.

    Operates on the full composed `text` column from pl_hcl parquets — the
    layer-specific columns aren't materialised at write time, only the
    composed string. Returns "" if the layer block is absent.
    """
    if not full_text:
        return ""
    m = _RGX_BLOCK[layer].search(full_text)
    return m.group(1) if m else ""


def compose_layers(parts: "Dict[str, str]", stage: str) -> str:
    """Re-emit ``<METADATA>...</METADATA>\\n<INSTRUCTION>...</INSTRUCTION>...`` etc.

    `parts` is a mapping of layer code → inner text (no tags). Layers with
    empty inner are dropped. Order matches Phase 1's composition rule.
    """
    order = ("M", "I") if stage == "stage1" else ("M", "I", "R")
    blocks = []
    for ll in order:
        v = parts.get(ll, "")
        if v:
            blocks.append(wrap(ll, v))
    return "\n".join(blocks)


def wrap(layer: str, inner: str) -> str:
    if not inner:
        return ""
    if layer == "M":
        return f"<METADATA>\n{inner}\n</METADATA>"
    if layer == "I":
        return f"<INSTRUCTION>\n{inner}\n</INSTRUCTION>"
    if layer == "R":
        return f"<RESOURCE>\n{inner}\n</RESOURCE>"
    raise ValueError(layer)


def est_tokens(text: str) -> int:
    if not text:
        return 0
    return int((len(text) + 3) // 4)


# ---------------------------------------------------------------------------
# Scorers — assign a "rewritability" weight to each line for each label.
# ---------------------------------------------------------------------------

# T3a: lines worth flipping behavior on. We want logic / control flow /
# constants / function calls — places where a small edit changes runtime
# behavior using identifiers that already exist.
_T3A_PATTERNS = [
    re.compile(r"\b(if|elif|else|while|for|switch|case|match)\b"),
    re.compile(r"(==|!=|<=|>=|[<>])"),
    re.compile(r"\b(return|break|continue|raise|throw|yield)\b"),
    re.compile(r"\b(true|false|True|False|null|None)\b"),
    re.compile(r"\b\d+\.?\d*\b"),                # numeric literal
    re.compile(r"\w+\([^()]{0,80}\)"),            # function call w/ args
    re.compile(r"=[^=]"),                         # assignment
]

# T3c: lines that introduce or reference *identifiers from outside* the
# current artifact — the prime targets for fabricated-identifier injection.
_T3C_PATTERNS = [
    re.compile(r"^\s*(import|from)\s+[\w.]+", re.MULTILINE),       # py import
    re.compile(r"\brequire\s*\(\s*['\"][^'\"]+['\"]\s*\)"),          # js require
    re.compile(r"\bfrom\s+['\"][^'\"]+['\"]\s+import"),             # js import
    re.compile(r"\b(npm|yarn|pnpm)\s+(install|add)\s+\S+"),
    re.compile(r"\bpip\s+install\s+\S+"),
    re.compile(r'"@[\w-]+/[\w-]+"\s*:\s*"[^"]+"'),                 # npm scoped dep
    re.compile(r'"[\w-]+"\s*:\s*"\^?\d'),                          # generic dep
    re.compile(r"^\s*[\w-]+==[\w.\-+]+", re.MULTILINE),             # pip pin
    re.compile(r"\b[A-Z_][A-Z0-9_]{2,}\b"),                         # ENV_VAR
    re.compile(r"--[a-z][\w-]+"),                                   # CLI flag
    re.compile(r"[\w./-]+\.(yaml|yml|json|toml|cfg|ini|env)\b"),    # config path
    re.compile(r"https?://[\w./?=&%-]+"),                           # URL
]


def _score_line(line: str, patterns: Sequence[re.Pattern]) -> int:
    if not line.strip():
        return 0
    s = 0
    for p in patterns:
        s += len(p.findall(line))
    return s


def score_t3a(line: str) -> int:
    return _score_line(line, _T3A_PATTERNS)


def score_t3c(line: str) -> int:
    return _score_line(line, _T3C_PATTERNS)


# ---------------------------------------------------------------------------
# Span sampling
# ---------------------------------------------------------------------------

@dataclass
class Span:
    start_line: int
    end_line: int                # exclusive
    text: str
    score: float = 0.0


def _seeded_rng(*parts: str) -> random.Random:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    return random.Random(seed)


def sample_spans(
    text: str,
    scorer: Callable[[str], int],
    k: int,
    span_lines: int = 8,
    min_lines_per_span: int = 4,
    max_lines_per_span: int = 12,
    seed_parts: Tuple[str, ...] = (),
) -> List[Span]:
    """Pick `k` non-overlapping spans of ~span_lines each, weighted by scorer.

    * Lines aligned (no mid-line cuts).
    * Deterministic given the same `seed_parts`.
    * If text has < k * min_lines_per_span lines, returns up to as many spans
      as fit, possibly fewer than k.
    """
    if not text:
        return []
    lines = text.split("\n")
    n = len(lines)
    if n < min_lines_per_span:
        return [Span(0, n, text, score=float(sum(scorer(ll) for ll in lines)))]

    rng = _seeded_rng("sample_spans", str(k), str(span_lines), *seed_parts)

    # Pre-score; uniform floor of 1 keeps low-score lines reachable.
    line_scores = [scorer(ll) + 1 for ll in lines]

    # Candidate spans = sliding windows of `span_lines`, scored by sum of line
    # scores. We then sample `k` non-overlapping windows weighted by score.
    L = max(min_lines_per_span, min(max_lines_per_span, span_lines))
    candidates: List[Tuple[int, int, float]] = []
    if n <= L:
        candidates.append((0, n, float(sum(line_scores))))
    else:
        cum = [0]
        for s in line_scores:
            cum.append(cum[-1] + s)
        for start in range(0, n - L + 1):
            score = float(cum[start + L] - cum[start])
            candidates.append((start, start + L, score))

    chosen: List[Span] = []
    used: List[Tuple[int, int]] = []
    # Weighted sampling without replacement, with non-overlap constraint.
    available = list(candidates)
    while available and len(chosen) < k:
        weights = [c[2] for c in available]
        if sum(weights) <= 0:
            pick = rng.choice(available)
        else:
            pick = rng.choices(available, weights=weights, k=1)[0]
        s, e, sc = pick
        chosen.append(
            Span(start_line=s, end_line=e, text="\n".join(lines[s:e]), score=sc)
        )
        used.append((s, e))
        available = [c for c in available if c[1] <= s or c[0] >= e]
    chosen.sort(key=lambda x: x.start_line)
    return chosen


def splice_spans(
    text: str,
    spans: Sequence[Span],
    rewrites: Sequence[str],
) -> str:
    """Replace each span in `text` with the corresponding rewrite (line-aligned).

    Spans must be sorted ascending and non-overlapping (sample_spans guarantees
    this). `rewrites` and `spans` must be parallel; entries with empty rewrite
    are skipped (original kept).
    """
    if not spans:
        return text
    if len(rewrites) != len(spans):
        raise ValueError(
            f"splice mismatch: {len(spans)} spans vs {len(rewrites)} rewrites"
        )
    lines = text.split("\n")
    out: List[str] = []
    cursor = 0
    for sp, rw in zip(spans, rewrites):
        if sp.start_line < cursor:
            raise ValueError(f"overlapping span at line {sp.start_line}")
        out.extend(lines[cursor:sp.start_line])
        if rw and rw.strip():
            out.append(rw.rstrip("\n"))
        else:
            out.extend(lines[sp.start_line:sp.end_line])
        cursor = sp.end_line
    out.extend(lines[cursor:])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Multi-span prompt format — single LLM call returns K rewrites delimited.
# ---------------------------------------------------------------------------

SPAN_DELIM = "===SPAN==="
REWRITE_DELIM = "===REWRITE==="


def build_spans_block(spans: Sequence[Span]) -> str:
    blocks = []
    for i, sp in enumerate(spans):
        blocks.append(f"{SPAN_DELIM} {i + 1} {SPAN_DELIM}\n{sp.text}")
    return "\n".join(blocks)


_REWRITE_SPLIT_RGX = re.compile(
    r"={2,}\s*(?:\d+\s*)?REWRITE(?:\s*\d+)?\s*={2,}(?:\s*\d+\s*={2,})?",
    re.IGNORECASE,
)


def parse_rewrites(model_output: str, k: int) -> List[str]:
    """Pull K rewrites out of the model's response. Tolerant of stray prose.

    Splits on `===REWRITE===` markers (case-insensitive, accepts a number
    between markers like `===REWRITE=== 1 ===`). The first chunk before the
    first marker is dropped (typically empty or a model preamble). The
    remaining chunks are stripped of fences/preamble; up to K non-empty
    rewrites are returned, padded with "" if the model produced fewer.
    """
    if not model_output:
        return [""] * k
    parts = _REWRITE_SPLIT_RGX.split(model_output)
    # Drop the pre-first-marker chunk (preamble or empty).
    chunks = parts[1:] if len(parts) > 1 else parts
    cleaned: List[str] = []
    for p in chunks:
        p = _strip_preamble(p)
        if p.strip():
            cleaned.append(p.strip())
    if len(cleaned) >= k:
        return cleaned[:k]
    while len(cleaned) < k:
        cleaned.append("")
    return cleaned


_PREAMBLE_RGX = re.compile(
    r"^\s*(here(?:'s| is| are)|note:|sure[,!]|this is|the following|"
    r"i (?:cannot|can't|won't)|sorry[,]?|okay[,!]?|let me)",
    re.IGNORECASE,
)
_FENCE_RGX = re.compile(r"^```[\w-]*\s*\n|\n```\s*$", re.MULTILINE)


def _strip_preamble(text: str) -> str:
    if not text:
        return text
    text = _FENCE_RGX.sub("", text)
    # Drop leading lines that look like commentary.
    lines = text.splitlines()
    while lines and _PREAMBLE_RGX.match(lines[0] or ""):
        lines.pop(0)
    return "\n".join(lines).strip()
