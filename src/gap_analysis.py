from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List


@dataclass
class GapItem:
    skill: str
    category: str
    priority: str
    reason: str


class GapAnalyzer:
    def __init__(self, lexical_threshold: float = 0.82):
        self.lexical_threshold = lexical_threshold

    def analyze(
        self,
        resume_skills: Dict[str, List[str]],
        jd_skills: Dict[str, List[str]],
        jd_text: str,
    ) -> List[GapItem]:
        resume_all = self._flatten(resume_skills)
        gaps: List[GapItem] = []

        for category, skills in jd_skills.items():
            for skill in skills:
                if not self._has_match(skill, resume_all):
                    priority = self._priority(skill, jd_text)
                    gaps.append(
                        GapItem(
                            skill=skill,
                            category=category,
                            priority=priority,
                            reason=f"Mentioned in JD but no close resume match for '{skill}'.",
                        )
                    )

        return sorted(gaps, key=lambda g: (g.priority != "critical", g.category, g.skill))

    def suggest_resume_improvements(self, gaps: List[GapItem]) -> List[str]:
        suggestions: List[str] = []
        for gap in gaps:
            if gap.category == "technical":
                suggestions.append(
                    f"Add evidence for {gap.skill}: include a project bullet with tool, task, and measurable outcome."
                )
            elif gap.category == "certifications":
                suggestions.append(
                    f"If applicable, add or pursue {gap.skill} and list it in a Certifications section."
                )
            else:
                suggestions.append(
                    f"Strengthen {gap.skill} signals by adding concise STAR-style bullets in Experience."
                )

        return suggestions[:12]

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
        freq = jd_text.count(skill)
        if freq >= 2:
            return "critical"
        if freq == 1:
            return "important"
        return "nice-to-have"
