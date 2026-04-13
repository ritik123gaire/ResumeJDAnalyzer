from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


DEFAULT_SKILL_DB = {
    "technical": {
        "python",
        "java",
        "sql",
        "pandas",
        "numpy",
        "machine learning",
        "deep learning",
        "nlp",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "tableau",
        "power bi",
        "excel",
        "spark",
        "hadoop",
        "git",
        "linux",
        "javascript",
        "react",
    },
    "soft": {
        "communication",
        "leadership",
        "teamwork",
        "problem solving",
        "critical thinking",
        "collaboration",
        "adaptability",
        "time management",
    },
    "certifications": {
        "aws certified",
        "pmp",
        "cfa",
        "scrum master",
        "google cloud certified",
        "azure fundamentals",
    },
}

ALIASES = {
    "js": "javascript",
    "ml": "machine learning",
    "nlp": "nlp",
    "scikitlearn": "scikit-learn",
    "powerbi": "power bi",
    "k8s": "kubernetes",
}


@dataclass
class SkillExtractionResult:
    technical: List[str]
    soft: List[str]
    certifications: List[str]

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "technical": self.technical,
            "soft": self.soft,
            "certifications": self.certifications,
        }


class SkillExtractor:
    def __init__(self, skill_db: Dict[str, Set[str]] | None = None):
        self.skill_db = skill_db or DEFAULT_SKILL_DB

    def extract(self, text: str) -> SkillExtractionResult:
        normalized = self._normalize_text(text)
        tokens = set(normalized.split())

        technical = self._find_matches(normalized, tokens, self.skill_db["technical"])
        soft = self._find_matches(normalized, tokens, self.skill_db["soft"])
        certs = self._find_matches(normalized, tokens, self.skill_db["certifications"])

        certs.extend(self._regex_certifications(normalized))

        return SkillExtractionResult(
            technical=sorted(set(technical)),
            soft=sorted(set(soft)),
            certifications=sorted(set(certs)),
        )

    def _find_matches(self, normalized_text: str, tokens: Set[str], vocabulary: Iterable[str]) -> List[str]:
        hits = []
        for skill in vocabulary:
            skill_norm = self._normalize_alias(skill)
            if " " in skill_norm:
                if skill_norm in normalized_text:
                    hits.append(skill_norm)
            elif skill_norm in tokens:
                hits.append(skill_norm)
        return hits

    @staticmethod
    def _regex_certifications(text: str) -> List[str]:
        patterns = [
            r"\baws (certified|solutions architect|developer)\b",
            r"\bgoogle (professional|associate) cloud\b",
            r"\b(certified )?scrum master\b",
            r"\bpmp\b",
        ]
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text))
        return [m if isinstance(m, str) else " ".join([x for x in m if x]) for m in matches]

    def _normalize_text(self, text: str) -> str:
        lowered = re.sub(r"[^a-zA-Z0-9+\-./ ]", " ", text.lower())
        lowered = re.sub(r"\s+", " ", lowered).strip()
        words = [self._normalize_alias(token) for token in lowered.split()]
        return " ".join(words)

    @staticmethod
    def _normalize_alias(token: str) -> str:
        token = token.strip().lower()
        return ALIASES.get(token, token)
