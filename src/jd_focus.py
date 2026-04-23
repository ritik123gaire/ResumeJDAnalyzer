from __future__ import annotations

import re
from typing import Dict, List, Tuple


# Lines that often start the *role-relevant* part of a posting (normalized lower text).
_START_LINE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^\s*job\s+description", re.I),
    re.compile(r"^\s*role\s+overview", re.I),
    re.compile(r"^\s*position\s+summary", re.I),
    re.compile(r"^\s*responsibilit", re.I),
    re.compile(r"^\s*key\s+responsibilit", re.I),
    re.compile(r"^\s*what\s+you\s+will", re.I),
    re.compile(r"^\s*in\s+this\s+(role|position)", re.I),
    re.compile(r"^\s*you\s+will\s*[:]", re.I),
    re.compile(r"^\s*minimum\s+qualifications", re.I),
    re.compile(r"^\s*required\s+qualifications", re.I),
    re.compile(r"^\s*qualifications\s*$", re.I),
    re.compile(r"^\s*preferred\s+qualifications", re.I),
    re.compile(r"^\s*basic\s+qualifications", re.I),
    re.compile(r"^\s*kla\s+is\s+seeking", re.I),
    re.compile(r"^\s*we\s+are\s+seeking", re.I),
    re.compile(r"^\s*seeking\s+a\s+motivated", re.I),
]

# Lines that usually begin boilerplate / legal / comp blocks (end of role-relevant region).
_END_LINE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^base\s+pay", re.I),
    re.compile(r"^compensation", re.I),
    re.compile(r"^salary\s+range", re.I),
    re.compile(r"^pay\s+range", re.I),
    re.compile(r"^pay\s+rate", re.I),
    re.compile(r"^currency", re.I),
    re.compile(r"^primary\s+location", re.I),
    re.compile(r"^location\s*:", re.I),
    re.compile(r"^your\s+location", re.I),
    re.compile(r"^remote\s+eligible", re.I),
    re.compile(r"^total\s+rewards", re.I),
    re.compile(r"^benefits\s+package", re.I),
    re.compile(r"^our\s+benefits", re.I),
    re.compile(r"^employee\s+travel\s+credits", re.I),
    re.compile(r"^equal\s+opportunity", re.I),
    re.compile(r"^equal\s+opportunity\s+employer", re.I),
    re.compile(r"^our\s+commitment", re.I),
    re.compile(r"^commitment\s+to\s+inclusion", re.I),
    re.compile(r"^inclusion\s*&\s*belonging", re.I),
    re.compile(r"^how\s+we'?ll\s+take\s+care\s+of\s+you", re.I),
    re.compile(r"^kla\s+is\s+proud\s+to\s+be\s+an\s+equal", re.I),
    re.compile(r"^fraudulent\s+job", re.I),
    re.compile(r"^be\s+aware\s+of\s+potentially\s+fraudulent", re.I),
    re.compile(r"^privacy", re.I),
    re.compile(r"^we\s+take\s+your\s+privacy", re.I),
    re.compile(r"^disability\s+inclusive", re.I),
    re.compile(r"^reasonable\s+accommodation", re.I),
    re.compile(r"^work\s+authorization", re.I),
    re.compile(r"^apply\s+now\s+at", re.I),
]

_END_LINE_CONTAINS: List[re.Pattern[str]] = [
    re.compile(r"work\s+authorization", re.I),
    re.compile(r"reasonable\s+accommodation", re.I),
    re.compile(r"equal\s+opportunity", re.I),
    re.compile(r"pay\s+range", re.I),
    re.compile(r"pay\s+rate", re.I),
    re.compile(r"employee\s+travel\s+credits", re.I),
]


def focus_job_description(normalized_jd: str, min_chars: int = 180) -> Tuple[str, Dict[str, object]]:
    """
    Keep the parts of a JD that usually matter for fit (role + requirements),
    and drop common corporate / HR / legal tail sections.

    `normalized_jd` should already be lowercased / cleaned like `TextPreprocessor.normalize_text`.
    """
    meta: Dict[str, object] = {
        "used_full_text": False,
        "trim_applied": False,
        "start_line_index": None,
        "end_line_index": None,
        "focused_chars": 0,
        "full_chars": len(normalized_jd or ""),
        "note": "",
    }
    text = (normalized_jd or "").strip()
    if not text:
        return "", meta

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        meta["used_full_text"] = True
        return text, meta

    start_idx = 0
    for i, line in enumerate(lines):
        if any(p.search(line) for p in _START_LINE_PATTERNS):
            start_idx = i
            meta["start_line_index"] = i
            break

    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        line = lines[j]
        if any(p.search(line) for p in _END_LINE_PATTERNS):
            end_idx = j
            meta["end_line_index"] = j
            break
        if any(p.search(line) for p in _END_LINE_CONTAINS):
            end_idx = j
            meta["end_line_index"] = j
            break

    focused_lines = lines[start_idx:end_idx]
    focused = "\n".join(focused_lines).strip()

    if len(focused) < min_chars:
        meta["used_full_text"] = True
        meta["note"] = "Focused window was too short; fell back to full posting text."
        return text, meta

    meta["trim_applied"] = (start_idx > 0) or (end_idx < len(lines))
    if not meta["trim_applied"]:
        meta["note"] = "No clear role/requirements headers found; using full posting (scores may be diluted by corporate boilerplate)."

    meta["focused_chars"] = len(focused)
    return focused, meta
