from __future__ import annotations

import re
from typing import Dict, List


def run_resume_applicant_checks(raw_resume: str, normalized_resume: str, sections: Dict[str, str]) -> Dict[str, object]:
    """
    Lightweight checks that help real applicants (ATS-ish + clarity), not hiring decisions.
    """
    tips: List[str] = []
    raw = raw_resume or ""
    norm = normalized_resume or ""

    word_count = len(norm.split())
    if word_count < 120:
        tips.append("Resume text looks short for a full CV; ensure all roles, dates, and impact bullets are included.")
    if word_count > 900:
        tips.append("Resume is very long; consider tightening to the most relevant roles for this posting.")

    if not re.search(r"\b20\d{2}\b", norm):
        tips.append("Add clear year ranges (e.g., 2021–2024) so readers and parsers can scan your timeline quickly.")

    if not sections.get("experience") and not sections.get("skills"):
        tips.append(
            "Use standard section headers like 'Experience' and 'Skills' so tools (and recruiters) can find key content."
        )

    bullet_like = norm.count("•") + norm.count("- ") + norm.count("– ")
    if bullet_like < 3 and word_count > 200:
        tips.append("Use bullet points for accomplishments; dense paragraphs are harder to skim and parse.")

    if len(re.findall(r"\b[a-z]{20,}\b", norm)) > 5:
        tips.append("Break up very long lines or URLs; some parsers struggle with dense unbroken text.")

    if not tips:
        tips.append("Formatting looks reasonable; focus next on aligning bullets to the JD requirements section.")

    return {
        "word_count": word_count,
        "section_keys_found": sorted(k for k, v in (sections or {}).items() if (v or "").strip()),
        "tips": tips[:8],
        "disclaimer": "These checks are guidance only; employer ATS behavior varies.",
    }
