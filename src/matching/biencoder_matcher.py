from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class BiEncoderResult:
    score: float
    best_skill_pairs: List[Tuple[str, str, float]]


class BiEncoderMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, resume_text: str, jd_text: str) -> float:
        emb = self.model.encode([resume_text, jd_text], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))

    def skill_level_similarity(
        self,
        resume_skills: List[str],
        jd_skills: List[str],
        threshold: float = 0.45,
        top_k: int = 20,
    ) -> BiEncoderResult:
        if not resume_skills or not jd_skills:
            return BiEncoderResult(score=0.0, best_skill_pairs=[])

        resume_emb = self.model.encode(resume_skills, normalize_embeddings=True)
        jd_emb = self.model.encode(jd_skills, normalize_embeddings=True)

        sim = np.matmul(jd_emb, resume_emb.T)

        best_pairs: List[Tuple[str, str, float]] = []
        best_scores = []
        for jd_idx, row in enumerate(sim):
            best_idx = int(np.argmax(row))
            score = float(row[best_idx])
            best_scores.append(score)
            if score >= threshold:
                best_pairs.append((jd_skills[jd_idx], resume_skills[best_idx], score))

        overall = float(np.mean(best_scores)) if best_scores else 0.0
        best_pairs = sorted(best_pairs, key=lambda x: x[2], reverse=True)[:top_k]
        return BiEncoderResult(score=overall, best_skill_pairs=best_pairs)
