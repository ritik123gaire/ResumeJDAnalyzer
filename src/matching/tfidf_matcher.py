from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MatchResult:
    score: float
    top_resume_terms: List[Tuple[str, float]]
    top_jd_terms: List[Tuple[str, float]]


class TFIDFMatcher:
    def __init__(self, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")

    def score(self, resume_text: str, jd_text: str) -> MatchResult:
        matrix = self.vectorizer.fit_transform([resume_text, jd_text])
        similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])

        feature_names = self.vectorizer.get_feature_names_out()
        resume_weights = matrix[0].toarray().ravel()
        jd_weights = matrix[1].toarray().ravel()

        top_resume = self._top_weighted_terms(feature_names, resume_weights)
        top_jd = self._top_weighted_terms(feature_names, jd_weights)

        return MatchResult(score=similarity, top_resume_terms=top_resume, top_jd_terms=top_jd)

    @staticmethod
    def skill_gap_terms(match_result: MatchResult, top_k: int = 12) -> List[str]:
        resume_terms = {t for t, _ in match_result.top_resume_terms}
        gaps = [t for t, _ in match_result.top_jd_terms if t not in resume_terms]
        return gaps[:top_k]

    @staticmethod
    def _top_weighted_terms(feature_names: np.ndarray, weights: np.ndarray, top_n: int = 20) -> List[Tuple[str, float]]:
        if not len(feature_names):
            return []
        idx = np.argsort(weights)[::-1][:top_n]
        return [(str(feature_names[i]), float(weights[i])) for i in idx if weights[i] > 0]

    def rank_candidates(self, resume_text: str, jd_texts: Dict[str, str]) -> List[Tuple[str, float]]:
        ids = list(jd_texts.keys())
        corpus = [resume_text] + [jd_texts[jid] for jid in ids]
        matrix = self.vectorizer.fit_transform(corpus)
        resume_vec = matrix[0:1]
        jd_vecs = matrix[1:]
        scores = cosine_similarity(resume_vec, jd_vecs)[0]
        ranked = sorted(zip(ids, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [(jid, float(score)) for jid, score in ranked]
