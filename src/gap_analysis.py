from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Dict, List


@dataclass
class GapItem:
    skill: str
    category: str
    priority: str
    reason: str
    jd_evidence: str
    requirement_level: str


class GapAnalyzer:
    def __init__(self, lexical_threshold: float = 0.82):
        self.lexical_threshold = lexical_threshold

    def analyze(
        self,
        resume_skills: Dict[str, List[str]],
        jd_skills: Dict[str, List[str]],
        jd_text: str,
        resume_text: str,
        resume_sections: Dict[str, str] | None = None,
    ) -> List[GapItem]:
        resume_all = self._flatten(resume_skills)
        sections = resume_sections or {}
        gaps: List[GapItem] = []
        seen_skills = set()

        for category, skills in jd_skills.items():
            for skill in skills:
                if not self._has_match(skill, resume_all):
                    priority = self._priority(skill, jd_text)
                    evidence = self.skill_evidence(skill, resume_text, sections)
                    jd_snip = self._jd_snippet(jd_text, skill)
                    req_level = self._requirement_level(priority)
                    gap = GapItem(
                        skill=skill,
                        category=category,
                        priority=priority,
                        reason=(
                            f"Mentioned in JD but no close resume match for '{skill}'. "
                            f"Resume evidence confidence: {evidence['confidence']}."
                        ),
                        jd_evidence=jd_snip,
                        requirement_level=req_level,
                    )
                    key = gap.skill
                    if key not in seen_skills:
                        gaps.append(gap)
                        seen_skills.add(key)

        for inferred_skill, inferred_priority in self._infer_requirement_terms(jd_text):
            if self._has_match(inferred_skill, resume_all):
                continue
            if re.search(rf"\b{re.escape(inferred_skill)}\b", resume_text.lower()):
                continue

            category = self._infer_category(inferred_skill)
            evidence = self.skill_evidence(inferred_skill, resume_text, sections)
            gap = GapItem(
                skill=inferred_skill,
                category=category,
                priority=inferred_priority,
                reason=(
                    f"Inferred from JD requirement phrasing but not found in resume text for '{inferred_skill}'. "
                    f"Resume evidence confidence: {evidence['confidence']}."
                ),
                jd_evidence=self._jd_snippet(jd_text, inferred_skill),
                requirement_level=self._requirement_level(inferred_priority),
            )
            key = gap.skill
            if key not in seen_skills:
                gaps.append(gap)
                seen_skills.add(key)

        return sorted(gaps, key=lambda g: (g.priority != "critical", g.category, g.skill))

    def suggest_resume_improvements(self, gaps: List[GapItem]) -> List[str]:
        suggestions: List[str] = []
        for gap in gaps:
            jd_hint = f' JD cites: "{gap.jd_evidence}"' if gap.jd_evidence else ""
            label = gap.requirement_level.replace("_", " ")
            if gap.category == "technical":
                suggestions.append(
                    f"[{label}] Add evidence for {gap.skill}: one bullet with tool + task + metric.{jd_hint}"
                )
            elif gap.category == "certifications":
                suggestions.append(
                    f"[{label}] If applicable, add or pursue {gap.skill} and list it under Certifications.{jd_hint}"
                )
            else:
                suggestions.append(
                    f"[{label}] Strengthen {gap.skill}: add a short STAR bullet in Experience tied to outcomes.{jd_hint}"
                )

        return suggestions[:12]

    @staticmethod
    def _requirement_level(priority: str) -> str:
        if priority == "critical":
            return "must_have"
        if priority == "important":
            return "preferred"
        return "mentioned"

    @staticmethod
    def _jd_snippet(jd_text: str, skill: str, radius: int = 140) -> str:
        if not jd_text or not skill:
            return ""
        lowered = jd_text.lower()
        needle = skill.lower()
        pos = lowered.find(needle)
        if pos < 0:
            return ""
        start = max(0, pos - radius)
        end = min(len(jd_text), pos + len(needle) + radius)
        snippet = jd_text[start:end].replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(jd_text):
            snippet = snippet + "…"
        return snippet[:320]

    def skill_evidence(self, skill: str, resume_text: str, resume_sections: Dict[str, str]) -> Dict[str, str | float]:
        """Estimate how strongly a resume demonstrates a skill."""
        section_weights = {
            "experience": 1.0,
            "projects": 0.9,
            "skills": 0.65,
            "summary": 0.45,
            "other": 0.5,
        }
        action_verbs = {"built", "designed", "implemented", "optimized", "automated", "delivered", "led", "deployed"}
        metric_patterns = [r"\b\d+%\b", r"\b\d+(\.\d+)?x\b", r"\$\d+", r"\b\d+\s*(users|clients|records|hours|days)\b"]

        skill_pattern = re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)
        metric_hits = sum(len(re.findall(pattern, resume_text.lower())) for pattern in metric_patterns)
        action_hits = sum(1 for verb in action_verbs if re.search(rf"\b{verb}\b", resume_text.lower()))

        weighted_presence = 0.0
        for section_name, section_text in resume_sections.items():
            if skill_pattern.search(section_text):
                weighted_presence += section_weights.get(section_name, 0.45)

        base = min(1.0, weighted_presence)
        evidence_score = max(0.0, min(1.0, base + min(action_hits, 2) * 0.1 + min(metric_hits, 2) * 0.1))

        if evidence_score >= 0.75:
            confidence = "high"
        elif evidence_score >= 0.45:
            confidence = "medium"
        else:
            confidence = "low"

        return {"score": evidence_score, "confidence": confidence}

    @staticmethod
    def _flatten(skills_dict: Dict[str, List[str]]) -> List[str]:
        items = []
        for values in skills_dict.values():
            items.extend(values)
        return list(set(items))

    def _has_match(self, skill: str, resume_skills: List[str]) -> bool:
        for r_skill in resume_skills:
            if skill == r_skill:
                return True
            ratio = SequenceMatcher(a=skill, b=r_skill).ratio()
            if ratio >= self.lexical_threshold:
                return True
        return False

    @staticmethod
    def _priority(skill: str, jd_text: str) -> str:
        lowered = jd_text.lower()
        freq = lowered.count(skill.lower())
        critical_context = re.search(
            rf"(must|required|mandatory|need|minimum).{{0,40}}\b{re.escape(skill.lower())}\b|\b{re.escape(skill.lower())}\b.{{0,40}}(must|required|mandatory|need|minimum)",
            lowered,
        )
        preferred_context = re.search(
            rf"(preferred|plus|nice to have).{{0,40}}\b{re.escape(skill.lower())}\b|\b{re.escape(skill.lower())}\b.{{0,40}}(preferred|plus|nice to have)",
            lowered,
        )
        if freq >= 2 or critical_context:
            return "critical"
        if freq == 1 or preferred_context:
            return "important"
        return "nice-to-have"

    @staticmethod
    def _infer_category(skill: str) -> str:
        cert_tokens = ("certified", "certification", "license", "pmp", "scrum", "aws", "azure", "gcp")
        soft_tokens = (
            "communication",
            "leadership",
            "collaboration",
            "teamwork",
            "stakeholder",
            "presentation",
            "problem solving",
            "critical thinking",
        )
        lowered = skill.lower()
        if any(t in lowered for t in cert_tokens):
            return "certifications"
        if any(t in lowered for t in soft_tokens):
            return "soft"
        return "technical"

    @staticmethod
    def _infer_requirement_terms(jd_text: str) -> List[tuple[str, str]]:
        if not jd_text:
            return []

        requirement_patterns = [
            (re.compile(r"(must|required|mandatory)\s+(?:have\s+)?([^.\n;:]{3,140})", re.I), "critical"),
            (re.compile(r"(experience with|proficient in|proficiency in|knowledge of|hands[- ]on with)\s+([^.\n;:]{3,140})", re.I), "important"),
            (re.compile(r"(preferred|nice to have|plus)\s+([^.\n;:]{3,140})", re.I), "important"),
        ]
        stop_phrases = {
            "a",
            "an",
            "the",
            "ability",
            "ability to",
            "strong",
            "strong ability",
            "experience",
            "knowledge",
            "skills",
            "skill",
            "background",
            "work",
            "team",
            "teams",
            "environment",
            "role",
        }
        terms: List[tuple[str, str]] = []
        seen = set()

        for pattern, priority in requirement_patterns:
            for match in pattern.finditer(jd_text):
                chunk = match.group(2).lower()
                parts: List[str] = []
                parenthetical = re.findall(r"\(([^)]+)\)", chunk)
                for group in parenthetical:
                    parts.extend(re.split(r",|/|\bor\b|\band\b", group))
                chunk = re.sub(r"\([^)]*\)", " ", chunk)
                parts.extend(re.split(r",|/|\bor\b|\band\b", chunk))
                for raw in parts:
                    term = re.sub(r"\s+", " ", raw).strip(" .:-()[]{}")
                    term = re.sub(
                        r"^(experience with|proficient in|proficiency in|knowledge of|hands[- ]on with|with|in|using|for|to)\s+",
                        "",
                        term,
                    )
                    term = term.strip(" .:-()[]{}")
                    if not term or term in stop_phrases:
                        continue
                    words = [w for w in term.split() if len(w) > 1]
                    if len(words) > 4:
                        continue
                    if not any(ch.isalpha() for ch in term):
                        continue
                    if all(w in stop_phrases for w in words):
                        continue
                    if not GapAnalyzer._is_plausible_requirement_term(term):
                        continue
                    if term in seen:
                        continue
                    seen.add(term)
                    terms.append((term, priority))

        return terms[:24]

    @staticmethod
    def _is_plausible_requirement_term(term: str) -> bool:
        lowered = term.lower()
        blocked_exact = {
            "qualifications",
            "minimum qualifications",
            "preferred qualifications",
            "responsibilities",
            "requirements",
            "experience",
            "knowledge",
            "skills",
            "inc",
            "llc",
            "ltd",
            "corp",
            "co",
        }
        blocked_contains = (
            "graduate degree",
            "undergraduate",
            "senior year",
            "currently enrolled",
            "gpa",
            "years of age",
            "must be",
            "ability to",
            "work alongside",
            "industry experts",
            "real-world",
            "related field",
            "academic record",
            "work authorization",
            "authorized to work",
            "cpt/opt",
            "stem extension",
        )
        if lowered in blocked_exact:
            return False
        if re.fullmatch(r"(inc|llc|ltd|corp|co)", lowered):
            return False
        if any(token in lowered for token in blocked_contains):
            return False

        words = lowered.split()
        if len(words) < 1 or len(words) > 3:
            return False
        if all(w in {"strong", "excellent", "good", "solid"} for w in words):
            return False

        if len(words) == 1 and len(lowered) <= 3:
            short_allowlist = {"ai", "ml", "nlp", "cv", "can", "sql", "c++", "c#"}
            if lowered not in short_allowlist:
                return False

        if "protocol" in lowered:
            allowed_protocols = {
                "can",
                "ethernet",
                "i2c",
                "spi",
                "uart",
                "usb",
                "tcp",
                "udp",
                "http",
                "https",
                "mqtt",
                "ble",
                "zigbee",
            }
            if not any(token in allowed_protocols for token in words):
                return False

        # Keep phrases that look like skills/tools/protocols, including symbols like C++.
        return bool(re.fullmatch(r"[a-z0-9][a-z0-9+\-#\. ]*[a-z0-9+#]", lowered))
