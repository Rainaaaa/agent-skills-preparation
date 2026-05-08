#!/usr/bin/env python3

import hashlib
import json
import re
from typing import Any, Dict, List, Sequence, Tuple


BLANK_LINE_RE = re.compile(r"\n{3,}")
WHITESPACE_RE = re.compile(r"[ \t]+")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [WHITESPACE_RE.sub(" ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    return BLANK_LINE_RE.sub("\n\n", text)


def truncate_text(text: str, limit: int) -> Tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    clipped = text[:limit].rstrip()
    suffix = "\n\n[TRUNCATED]"
    if len(clipped) + len(suffix) > limit:
        clipped = clipped[: max(0, limit - len(suffix))].rstrip()
    return clipped + suffix, True


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_nested(record: Dict[str, Any], path: Sequence[str], default: Any = "") -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def build_package_files_payload(
    package_files: Any,
    max_package_file_chars: int,
    max_concat_chars: int,
) -> Dict[str, Any]:
    cleaned_files: List[Dict[str, Any]] = []
    rendered_blocks: List[str] = []
    total_bytes = 0
    truncated_count = 0
    remaining_concat = max_concat_chars

    if not isinstance(package_files, list):
        package_files = []

    for raw_file in package_files:
        if not isinstance(raw_file, dict):
            continue
        path = clean_text(raw_file.get("path", ""))
        content = clean_text(raw_file.get("content", ""))
        content, clipped_here = truncate_text(content, max_package_file_chars)
        original_truncated = bool(raw_file.get("truncated", False))
        final_truncated = bool(clipped_here or original_truncated)
        if final_truncated:
            truncated_count += 1
        total_bytes += int(raw_file.get("size_bytes") or 0)
        cleaned = {
            "path": path,
            "content": content,
            "size_bytes": int(raw_file.get("size_bytes") or 0),
            "truncated": final_truncated,
        }
        cleaned_files.append(cleaned)

        if remaining_concat > 0 and (path or content):
            block = f"### {path or 'unknown'}\n{content}".strip()
            block, block_truncated = truncate_text(block, remaining_concat)
            rendered_blocks.append(block)
            remaining_concat -= len(block) + 2
            if block_truncated and not final_truncated:
                truncated_count += 1

    joined_text = "\n\n".join(rendered_blocks).strip()
    return {
        "package_files": cleaned_files,
        "package_files_text": joined_text,
        "package_files_count": len(cleaned_files),
        "package_files_total_bytes": total_bytes,
        "package_files_truncated_count": truncated_count,
        "package_file_paths": [item["path"] for item in cleaned_files if item["path"]],
    }


def build_full_text(normalized_row: Dict[str, Any], max_concat_chars: int) -> str:
    sections: List[str] = []
    name = clean_text(normalized_row.get("name", ""))
    if name:
        sections.append(f"# Skill Name\n{name}")

    first_layer_description = clean_text(normalized_row.get("first_layer_description", ""))
    if first_layer_description:
        sections.append(f"# First Layer Description\n{first_layer_description}")

    description_text = clean_text(normalized_row.get("description_text", ""))
    if description_text:
        sections.append(f"# Effective Description\n{description_text}")

    non_description = clean_text(normalized_row.get("skill_md_non_description", ""))
    if non_description:
        sections.append(f"# Remaining Skill Markdown\n{non_description}")

    package_files_text = clean_text(normalized_row.get("package_files_text", ""))
    if package_files_text:
        sections.append(f"# Package Files\n{package_files_text}")

    full_text = "\n\n".join(sections).strip()
    return truncate_text(full_text, max_concat_chars)[0]


def normalize_record_task(task: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    record, runtime = task
    skill_id = clean_text(record.get("skill_id", ""))
    name = clean_text(get_nested(record, ("first_layer", "name"), ""))
    first_layer_description = clean_text(get_nested(record, ("first_layer", "description"), ""))
    skill_md_description = clean_text(
        get_nested(record, ("skill_md", "description_section", "content"), "")
    )
    has_explicit_description_heading = bool(
        get_nested(record, ("skill_md", "description_section", "found"), False)
    )
    skill_md_non_description = clean_text(get_nested(record, ("skill_md", "non_description"), ""))

    if skill_md_description:
        description_text = skill_md_description
        description_source = "skill_md_heading"
    elif first_layer_description:
        description_text = first_layer_description
        description_source = "first_layer_description"
    else:
        description_text = ""
        description_source = "empty"

    package_payload = build_package_files_payload(
        record.get("package_files", []),
        max_package_file_chars=int(runtime["max_package_file_chars"]),
        max_concat_chars=int(runtime["max_concat_chars"]),
    )

    normalized_row: Dict[str, Any] = {
        "skill_id": skill_id,
        "name": name,
        "first_layer_description": first_layer_description,
        "skill_md_description": skill_md_description,
        "description_text": description_text,
        "description_source": description_source,
        "skill_md_non_description": skill_md_non_description,
        "has_explicit_description_heading": has_explicit_description_heading,
        "used_metadata_description_fallback": bool(
            description_source == "first_layer_description" and first_layer_description
        ),
        "package_files_count": int(package_payload["package_files_count"]),
        "package_files_total_bytes": int(package_payload["package_files_total_bytes"]),
        "package_files_truncated_count": int(package_payload["package_files_truncated_count"]),
        "package_file_paths_json": stable_json_dumps(package_payload["package_file_paths"]),
        "package_files_json": stable_json_dumps(package_payload["package_files"]),
        "package_files_text": package_payload["package_files_text"],
    }

    normalized_row["full_text"] = build_full_text(normalized_row, int(runtime["max_concat_chars"]))
    normalized_row["text_char_count"] = len(normalized_row["full_text"])
    normalized_row["record_sha1"] = hashlib.sha1(
        stable_json_dumps(normalized_row).encode("utf-8")
    ).hexdigest()
    return normalized_row


METADATA_OPEN = "<METADATA>"
METADATA_CLOSE = "</METADATA>"
INSTRUCTION_OPEN = "<INSTRUCTION>"
INSTRUCTION_CLOSE = "</INSTRUCTION>"
RESOURCE_OPEN = "<RESOURCE>"
RESOURCE_CLOSE = "</RESOURCE>"


# Rough chars-per-token used for sizing only. Calibrated against
# Foundation-Sec / Qwen tokenizers on this corpus: typical English-y text
# averages ~4.5 chars/token, code-heavy content closer to ~3.8. We use 4.0
# as a single conservative estimate for budgeting.
_EST_CHARS_PER_TOKEN = 4.0


def _est_tokens(text: str) -> int:
    if not text:
        return 0
    # ceil-div so non-empty strings always get at least 1 token.
    n = len(text)
    return int((n + _EST_CHARS_PER_TOKEN - 1) // _EST_CHARS_PER_TOKEN)


def build_metadata_section(name: str, description_text: str, max_chars: int) -> str:
    name_clean = clean_text(name)
    desc_clean = clean_text(description_text)
    parts: List[str] = []
    if name_clean:
        parts.append(f"name: {name_clean}")
    if desc_clean:
        parts.append(f"description: {desc_clean}")
    if not parts:
        return ""
    body, _ = truncate_text("\n".join(parts), max_chars)
    return f"{METADATA_OPEN}\n{body}\n{METADATA_CLOSE}"


def build_instruction_section(skill_md_non_description: str, max_chars: int) -> str:
    body = clean_text(skill_md_non_description)
    if not body:
        return ""
    body, _ = truncate_text(body, max_chars)
    return f"{INSTRUCTION_OPEN}\n{body}\n{INSTRUCTION_CLOSE}"


def build_resource_section(
    package_files_json: str,
    max_per_file_chars: int,
    max_total_chars: int,
) -> Tuple[str, int, int, int]:
    """Pack per-skill files into one RESOURCE section.

    Returns (section_text, files_included, files_dropped, files_truncated).
    Files are appended in input order. ``max_per_file_chars=0`` and
    ``max_total_chars=0`` mean "no cap" (collect everything). When a total cap
    is set, files are truncated then dropped once the budget is exhausted; each
    file's content is also independently capped at ``max_per_file_chars`` when
    that is positive.
    """
    try:
        files = json.loads(package_files_json or "[]")
    except (TypeError, ValueError):
        files = []

    unlimited = max_total_chars <= 0
    remaining = float("inf") if unlimited else max_total_chars
    blocks: List[str] = []
    files_included = 0
    files_dropped = 0
    files_truncated = 0
    sep_len = 2  # for the "\n\n" between blocks

    for raw in files:
        if not isinstance(raw, dict):
            continue
        path = clean_text(raw.get("path", ""))
        content = clean_text(raw.get("content", ""))
        if not path and not content:
            continue
        if not unlimited and remaining <= 0:
            files_dropped += 1
            continue

        clipped, was_truncated = truncate_text(content, max_per_file_chars)
        block = f"### {path or 'unknown'}\n{clipped}".strip()
        if unlimited:
            blocks.append(block)
            files_included += 1
            if was_truncated:
                files_truncated += 1
            continue
        budget = remaining if not blocks else remaining - sep_len
        if budget <= 0:
            files_dropped += 1
            continue
        block, block_truncated = truncate_text(block, int(budget))
        if not block:
            files_dropped += 1
            continue
        blocks.append(block)
        files_included += 1
        if was_truncated or block_truncated:
            files_truncated += 1
        remaining -= len(block) + (sep_len if len(blocks) > 1 else 0)

    if not blocks:
        return "", 0, files_dropped, 0
    body = "\n\n".join(blocks)
    return f"{RESOURCE_OPEN}\n{body}\n{RESOURCE_CLOSE}", files_included, files_dropped, files_truncated


def _make_sample(
    *,
    skill_id: str,
    stage: str,
    text: str,
    metadata: str,
    instruction: str,
    resource: str,
    n_files_in: int,
    n_files_dropped: int,
    n_files_truncated: int,
) -> Dict[str, Any]:
    sample_id = hashlib.sha1(f"{stage}|{skill_id}|{text[:128]}".encode("utf-8")).hexdigest()
    return {
        "sample_id": sample_id,
        "skill_id": skill_id,
        "stage": stage,
        "text": text,
        "metadata_text": metadata,
        "instruction_text": instruction,
        "resource_text": resource,
        "metadata_chars": len(metadata),
        "instruction_chars": len(instruction),
        "resource_chars": len(resource),
        "total_chars": len(text),
        "metadata_tokens_est": _est_tokens(metadata),
        "instruction_tokens_est": _est_tokens(instruction),
        "resource_tokens_est": _est_tokens(resource),
        "total_tokens_est": _est_tokens(text),
        "n_resource_files_included": n_files_in,
        "n_resource_files_dropped": n_files_dropped,
        "n_resource_files_truncated": n_files_truncated,
    }


def build_phase1_samples_task(task: Tuple[Dict[str, Any], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build up to two composed Phase 1 rows per skill (stage 1, stage 2).

    Stage 1 row (METADATA + INSTRUCTION, capped at stage1_max_tokens):
      <METADATA>...</METADATA>
      <INSTRUCTION>...</INSTRUCTION>

    Stage 2 row (METADATA + INSTRUCTION + RESOURCE, capped at stage2_max_tokens;
    only emitted when the skill has any package files):
      <METADATA>...</METADATA>
      <INSTRUCTION>...</INSTRUCTION>
      <RESOURCE>...</RESOURCE>

    A skill may emit 0, 1, or 2 rows depending on which budgets it fits.
    Per-section char/token estimates are recorded on every row.
    """
    normalized_row, config = task
    skill_id = normalized_row["skill_id"]

    metadata = build_metadata_section(
        normalized_row.get("name", ""),
        normalized_row.get("description_text", ""),
        max_chars=int(config.get("max_metadata_chars", 0)),
    )
    instruction = build_instruction_section(
        normalized_row.get("skill_md_non_description", ""),
        max_chars=int(config.get("max_instruction_chars", 0)),
    )
    resource, n_files_in, n_files_dropped, n_files_truncated = build_resource_section(
        normalized_row.get("package_files_json", "[]"),
        max_per_file_chars=int(config.get("max_resource_per_file_chars", 0)),
        max_total_chars=int(config.get("max_resource_total_chars", 0)),
    )

    stage1_max = int(config.get("stage1_max_tokens", 4096))
    stage2_max = int(config.get("stage2_max_tokens", 10240))

    samples: List[Dict[str, Any]] = []

    # Stage 1: METADATA + INSTRUCTION
    stage1_parts = [p for p in (metadata, instruction) if p]
    if stage1_parts:
        stage1_text = "\n".join(stage1_parts)
        if _est_tokens(stage1_text) <= stage1_max:
            samples.append(
                _make_sample(
                    skill_id=skill_id,
                    stage="stage1",
                    text=stage1_text,
                    metadata=metadata,
                    instruction=instruction,
                    resource="",
                    n_files_in=0,
                    n_files_dropped=0,
                    n_files_truncated=0,
                )
            )

    # Stage 2: METADATA + INSTRUCTION + RESOURCE (only if resource present)
    if resource:
        stage2_parts = [p for p in (metadata, instruction, resource) if p]
        stage2_text = "\n".join(stage2_parts)
        if _est_tokens(stage2_text) <= stage2_max:
            samples.append(
                _make_sample(
                    skill_id=skill_id,
                    stage="stage2",
                    text=stage2_text,
                    metadata=metadata,
                    instruction=instruction,
                    resource=resource,
                    n_files_in=n_files_in,
                    n_files_dropped=n_files_dropped,
                    n_files_truncated=n_files_truncated,
                )
            )

    return samples


def build_stage_row_task(task: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    normalized_row, runtime = task
    max_stage_chars = int(runtime["max_stage_chars"])

    stage_name = truncate_text(clean_text(normalized_row.get("name", "")), max_stage_chars)[0]
    stage_description = truncate_text(
        clean_text(normalized_row.get("description_text", "")), max_stage_chars
    )[0]
    stage_markdown = truncate_text(
        clean_text(normalized_row.get("skill_md_non_description", "")), max_stage_chars
    )[0]
    stage_files = truncate_text(
        clean_text(normalized_row.get("package_files_text", "")), max_stage_chars
    )[0]
    packed_skill = truncate_text(clean_text(normalized_row.get("full_text", "")), max_stage_chars)[0]

    return {
        "skill_id": normalized_row["skill_id"],
        "stage_name": stage_name,
        "stage_description": stage_description,
        "stage_markdown": stage_markdown,
        "stage_files": stage_files,
        "packed_skill": packed_skill,
        "stage_name_len": len(stage_name),
        "stage_description_len": len(stage_description),
        "stage_markdown_len": len(stage_markdown),
        "stage_files_len": len(stage_files),
        "packed_skill_len": len(packed_skill),
    }


def build_phase3_texts(
    skill_name: str,
    evidence_text: str,
    candidate_description: str,
    evidence_limit: int,
) -> Tuple[str, str]:
    evidence, _ = truncate_text(clean_text(evidence_text), evidence_limit)
    candidate = clean_text(candidate_description)
    user_prompt = (
        f"Skill name: {clean_text(skill_name)}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Candidate description:\n{candidate}\n\n"
        "Does the candidate description match the skill? Answer yes or no."
    ).strip()
    return user_prompt, evidence
