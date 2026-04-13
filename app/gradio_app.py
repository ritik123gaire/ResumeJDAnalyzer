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
) -> tuple[str, str, str, str]:
    resume_payload = resume_text.strip() or _read_uploaded_file(resume_file)
    jd_payload = jd_text.strip() or _read_uploaded_file(jd_file)

    if not resume_payload or not jd_payload:
        return (
            "Please provide both resume and job description (text or file).",
            "{}",
            "[]",
            "[]",
        )

    result = analyzer.analyze(resume_payload, jd_payload, approach=approach)

    score_pct = round(result.score * 100, 2)
    score_msg = f"Match score ({result.approach}): {score_pct}%"

    skills_payload: Dict[str, Dict[str, list[str]]] = {
        "resume": result.resume_skills,
        "job_description": result.jd_skills,
    }
    gaps = result.gaps
    suggestions = result.suggestions

    return (
        score_msg,
        json.dumps(skills_payload, indent=2),
        json.dumps(gaps, indent=2),
        json.dumps(suggestions, indent=2),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Resume-to-JD Matcher with Gap Analysis") as demo:
        gr.Markdown("# Resume-to-Job-Description Matcher with Gap Analysis")
        gr.Markdown(
            "Compare a resume against a job description using TF-IDF, bi-encoder, or cross-encoder, then inspect extracted skills, gaps, and targeted resume suggestions."
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

        run_btn.click(
            fn=analyze_resume_jd,
            inputs=[resume_text, jd_text, resume_file, jd_file, approach],
            outputs=[score_out, skills_out, gaps_out, suggestions_out],
        )

    return demo


if __name__ == "__main__":
    demo_app = build_demo()
    demo_app.queue().launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=7860)
