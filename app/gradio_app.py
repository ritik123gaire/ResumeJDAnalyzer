from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ResumeJDAnalyzer


analyzer = ResumeJDAnalyzer()


def _read_uploaded_file(file_obj: Any) -> str:
    if not file_obj:
        return ""

    file_path = getattr(file_obj, "name", None) or str(file_obj)
    path = Path(file_path)
    if not path.exists():
        return ""

    if path.suffix.lower() == ".pdf":
        return analyzer.preprocessor.parse_input(file_path=str(path))
    return path.read_text(encoding="utf-8", errors="ignore")


def analyze_resume_jd(
    resume_text: str,
    jd_text: str,
    resume_file: Any,
    jd_file: Any,
    approach: str,
) -> tuple[str, str, str, str, str, str]:
    resume_payload = resume_text.strip() or _read_uploaded_file(resume_file)
    jd_payload = jd_text.strip() or _read_uploaded_file(jd_file)

    if not resume_payload or not jd_payload:
        return (
            "Please provide both resume and job description (text or file).",
            "{}",
            "[]",
            "[]",
            "{}",
            "{}",
        )

    result = analyzer.analyze(resume_payload, jd_payload, approach=approach)

    score_pct = round(result.score * 100, 2)
    score_msg = f"Match score ({result.approach}): {score_pct}% | confidence: {result.confidence}"

    skills_payload: Dict[str, Dict[str, list[str]]] = {
        "resume": result.resume_skills,
        "job_description": result.jd_skills,
    }
    gaps = result.gaps
    suggestions = result.suggestions
    jd_skill_count = sum(len(v) for v in result.jd_skills.values())

    if not gaps:
        if jd_skill_count == 0:
            gaps = [
                {
                    "note": (
                        "No JD skills were detected from the current vocabulary, "
                        "so explicit gap items could not be generated."
                    ),
                    "next_step": (
                        "Paste the JD requirements/responsibilities section (not only company overview), "
                        "or extend `DEFAULT_SKILL_DB` in `src/skill_extraction.py`."
                    ),
                }
            ]
        else:
            gaps = [
                {
                    "note": (
                        "No explicit missing skills were detected from the extracted JD skills."
                    ),
                    "next_step": (
                        "Refine bullets with outcomes and metrics for the strongest JD requirements."
                    ),
                }
            ]

    if not suggestions:
        if jd_skill_count == 0:
            suggestions = [
                "Add role-specific keywords from the JD requirements section so the analyzer can map likely skill gaps.",
                "If this job uses niche tools, add them to `DEFAULT_SKILL_DB` in `src/skill_extraction.py` for better coverage.",
            ]
        else:
            suggestions = [
                "No major skill gaps detected; focus on stronger achievement bullets (action + metric) for the top JD requirements.",
                "Mirror the JD wording for core tools/skills where truthful to improve parser and recruiter readability.",
            ]

    help_payload = {
        "applicant_tips": result.applicant_tips,
        "jd_focus": result.jd_focus,
    }

    return (
        score_msg,
        json.dumps(skills_payload, indent=2),
        json.dumps(gaps, indent=2),
        json.dumps(suggestions, indent=2),
        json.dumps(result.score_breakdown, indent=2),
        json.dumps(help_payload, indent=2),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Resume-to-JD Matcher with Gap Analysis") as demo:
        gr.Markdown("# Resume-to-Job-Description Matcher with Gap Analysis")
        gr.Markdown(
            "Compare a resume against a job description using TF-IDF, bi-encoder, or cross-encoder. "
            "Scores emphasize the role + requirements slice of long postings when possible; see **Applicant help** for JD focus preview and resume formatting tips."
        )

        with gr.Row():
            with gr.Column():
                resume_text = gr.Textbox(label="Resume Text", lines=14, placeholder="Paste resume text here...")
                resume_file = gr.File(label="Or Upload Resume (.txt/.pdf)", file_count="single", file_types=[".txt", ".pdf"])
            with gr.Column():
                jd_text = gr.Textbox(label="Job Description Text", lines=14, placeholder="Paste job description text here...")
                jd_file = gr.File(label="Or Upload JD (.txt/.pdf)", file_count="single", file_types=[".txt", ".pdf"])

        approach = gr.Dropdown(
            label="Matching Approach",
            choices=["tfidf", "bi_encoder", "cross_encoder"],
            value="bi_encoder",
        )

        run_btn = gr.Button("Analyze Match", variant="primary")

        score_out = gr.Textbox(label="Match Score")
        skills_out = gr.Code(label="Extracted Skills", language="json")
        gaps_out = gr.Code(label="Skill Gaps", language="json")
        suggestions_out = gr.Code(label="Improvement Suggestions", language="json")
        breakdown_out = gr.Code(label="Score Breakdown", language="json")
        help_out = gr.Code(label="Applicant help (JD focus + resume tips)", language="json")

        run_btn.click(
            fn=analyze_resume_jd,
            inputs=[resume_text, jd_text, resume_file, jd_file, approach],
            outputs=[score_out, skills_out, gaps_out, suggestions_out, breakdown_out, help_out],
        )

    return demo


if __name__ == "__main__":
    demo_app = build_demo()
    demo_app.queue().launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=7860)
