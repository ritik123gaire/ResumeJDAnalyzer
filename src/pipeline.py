from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.gap_analysis import GapAnalyzer
from src.matching.biencoder_matcher import BiEncoderMatcher
from src.matching.crossencoder_matcher import CrossEncoderMatcher
from src.matching.tfidf_matcher import TFIDFMatcher
from src.preprocessing import TextPreprocessor
from src.skill_extraction import SkillExtractor


@dataclass
class AnalysisOutput:
    approach: str
    score: float
    resume_skills: Dict[str, List[str]]
    jd_skills: Dict[str, List[str]]
    gaps: List[Dict[str, str]]
    suggestions: List[str]


class ResumeJDAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.skill_extractor = SkillExtractor()
        self.gap_analyzer = GapAnalyzer()

        self.tfidf = TFIDFMatcher()
        self.bi_encoder: BiEncoderMatcher | None = None
        self.cross_encoder: CrossEncoderMatcher | None = None

    def analyze(
        self,
        resume_text: str,
        jd_text: str,
        approach: str,
    ) -> AnalysisOutput:
        p_resume = self.preprocessor.preprocess(resume_text)
        p_jd = self.preprocessor.preprocess(jd_text)

        resume_skills = self.skill_extractor.extract(p_resume.normalized_text).as_dict()
        jd_skills = self.skill_extractor.extract(p_jd.normalized_text).as_dict()

        score = self._score(approach, p_resume.normalized_text, p_jd.normalized_text, resume_skills, jd_skills)

        gap_items = self.gap_analyzer.analyze(resume_skills, jd_skills, p_jd.normalized_text)
        suggestions = self.gap_analyzer.suggest_resume_improvements(gap_items)

        return AnalysisOutput(
            approach=approach,
            score=score,
            resume_skills=resume_skills,
            jd_skills=jd_skills,
            gaps=[
                {
                    "skill": g.skill,
                    "category": g.category,
                    "priority": g.priority,
                    "reason": g.reason,
                }
                for g in gap_items
            ],
            suggestions=suggestions,
        )

    def _score(
        self,
        approach: str,
        resume_text: str,
        jd_text: str,
        resume_skills: Dict[str, List[str]],
        jd_skills: Dict[str, List[str]],
    ) -> float:
        if approach == "tfidf":
            return self.tfidf.score(resume_text, jd_text).score

        resume_tech = resume_skills.get("technical", [])
        jd_tech = jd_skills.get("technical", [])

        if approach == "bi_encoder":
            if self.bi_encoder is None:
                self.bi_encoder = BiEncoderMatcher()
            doc_score = self.bi_encoder.score(resume_text, jd_text)
            skill_score = self.bi_encoder.skill_level_similarity(resume_tech, jd_tech).score
            return max(0.0, min(1.0, 0.7 * doc_score + 0.3 * skill_score))

        if approach == "cross_encoder":
            if self.cross_encoder is None:
                self.cross_encoder = CrossEncoderMatcher()
            return self.cross_encoder.score(resume_text, jd_text)

        raise ValueError(f"Unknown approach: {approach}")
