from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.applicant_checks import run_resume_applicant_checks
from src.gap_analysis import GapAnalyzer
from src.jd_focus import focus_job_description
from src.matching.biencoder_matcher import BiEncoderMatcher
from src.matching.crossencoder_matcher import CrossEncoderMatcher
from src.matching.tfidf_matcher import TFIDFMatcher
from src.preprocessing import TextPreprocessor
from src.skill_extraction import SkillExtractor


@dataclass
class AnalysisOutput:
    approach: str
    score: float
    confidence: str
    resume_skills: Dict[str, List[str]]
    jd_skills: Dict[str, List[str]]
    gaps: List[Dict[str, str]]
    suggestions: List[str]
    score_breakdown: Dict[str, float]
    jd_focus: Dict[str, Any]
    applicant_tips: Dict[str, Any]


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

        jd_focused, jd_focus_meta = focus_job_description(p_jd.normalized_text)
        jd_for_skills = jd_focused if len(jd_focused.strip()) >= 80 else p_jd.normalized_text

        resume_skills = self.skill_extractor.extract(p_resume.normalized_text).as_dict()
        jd_skills = self.skill_extractor.extract(jd_for_skills).as_dict()

        score, breakdown = self._score(
            approach,
            p_resume.normalized_text,
            jd_focused,
            p_resume.sections,
            resume_skills,
            jd_skills,
        )
        breakdown = {**breakdown, "jd_focus_used_full_text": bool(jd_focus_meta.get("used_full_text"))}

        gap_items = self.gap_analyzer.analyze(
            resume_skills,
            jd_skills,
            jd_focused,
            p_resume.normalized_text,
            p_resume.sections,
        )
        suggestions = self.gap_analyzer.suggest_resume_improvements(gap_items)

        applicant_tips = run_resume_applicant_checks(
            resume_text,
            p_resume.normalized_text,
            p_resume.sections,
        )

        return AnalysisOutput(
            approach=approach,
            score=score,
            confidence=self._score_confidence(score),
            resume_skills=resume_skills,
            jd_skills=jd_skills,
            gaps=[
                {
                    "skill": g.skill,
                    "category": g.category,
                    "priority": g.priority,
                    "reason": g.reason,
                    "jd_evidence": g.jd_evidence,
                    "requirement_level": g.requirement_level,
                }
                for g in gap_items
            ],
            suggestions=suggestions,
            score_breakdown=breakdown,
            jd_focus={
                "note": "Matching and JD skills use a focused slice of the posting (role + requirements) when detected.",
                "meta": jd_focus_meta,
                "preview": (jd_focused[:900] + "…") if len(jd_focused) > 900 else jd_focused,
            },
            applicant_tips=applicant_tips,
        )

    def _score(
        self,
        approach: str,
        resume_text: str,
        jd_text: str,
        resume_sections: Dict[str, str],
        resume_skills: Dict[str, List[str]],
        jd_skills: Dict[str, List[str]],
    ) -> Tuple[float, Dict[str, float]]:
        section_score = self._section_weighted_overlap(resume_sections, jd_text)
        if approach == "tfidf":
            doc_score = self.tfidf.score(resume_text, jd_text).score
            final_score = max(0.0, min(1.0, 0.8 * doc_score + 0.2 * section_score))
            return final_score, {"document_score": doc_score, "section_score": section_score}

        resume_tech = resume_skills.get("technical", [])
        jd_tech = jd_skills.get("technical", [])

        if approach == "bi_encoder":
            if self.bi_encoder is None:
                self.bi_encoder = BiEncoderMatcher()
            doc_score = self.bi_encoder.score(resume_text, jd_text)
            skill_score = self.bi_encoder.skill_level_similarity(resume_tech, jd_tech).score
            final_score = max(0.0, min(1.0, 0.6 * doc_score + 0.25 * skill_score + 0.15 * section_score))
            return final_score, {
                "document_score": doc_score,
                "skill_alignment_score": skill_score,
                "section_score": section_score,
            }

        if approach == "cross_encoder":
            if self.cross_encoder is None:
                self.cross_encoder = CrossEncoderMatcher()
            doc_score = self.cross_encoder.score(resume_text, jd_text)
            final_score = max(0.0, min(1.0, 0.85 * doc_score + 0.15 * section_score))
            return final_score, {"document_score": doc_score, "section_score": section_score}

        raise ValueError(f"Unknown approach: {approach}")

    @staticmethod
    def _score_confidence(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    @staticmethod
    def _section_weighted_overlap(resume_sections: Dict[str, str], jd_text: str) -> float:
        if not resume_sections:
            return 0.0

        section_weights = {
            "experience": 1.0,
            "skills": 0.85,
            "summary": 0.5,
            "education": 0.35,
            "certifications": 0.7,
            "other": 0.4,
        }
        jd_tokens = set(jd_text.split())
        if not jd_tokens:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0
        for section_name, text in resume_sections.items():
            tokens = set(text.split())
            overlap = len(tokens & jd_tokens) / len(jd_tokens) if tokens else 0.0
            weight = section_weights.get(section_name, 0.4)
            weighted_sum += overlap * weight
            weight_total += weight

        return weighted_sum / weight_total if weight_total else 0.0
