from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from PyPDF2 import PdfReader


SECTION_PATTERNS = {
    "summary": r"\b(summary|profile|objective)\b",
    "experience": r"\b(experience|employment|work history|professional experience)\b",
    "skills": r"\b(skills|technical skills|core competencies|technologies)\b",
    "education": r"\b(education|academic background|qualifications)\b",
    "certifications": r"\b(certifications|licenses)\b",
}


@dataclass
class ProcessedDocument:
    raw_text: str
    normalized_text: str
    sections: Dict[str, str]


class TextPreprocessor:
    """Handles resume/JD parsing, cleanup, and section segmentation."""

    def parse_input(self, text: str | None = None, file_path: str | None = None) -> str:
        if text and text.strip():
            return text

        if not file_path:
            raise ValueError("Either text or file_path must be provided.")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        if path.suffix.lower() == ".pdf":
            return self._read_pdf(file_path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def preprocess(self, text: str) -> ProcessedDocument:
        normalized = self.normalize_text(text)
        sections = self.segment_sections(normalized)
        return ProcessedDocument(raw_text=text, normalized_text=normalized, sections=sections)

    def _read_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\b\S+@\S+\.\S+\b", " ", text)
        text = re.sub(r"\+?\d[\d\-() ]{7,}\d", " ", text)
        text = re.sub(r"[^\S\r\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip().lower()

    def segment_sections(self, text: str) -> Dict[str, str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections: Dict[str, str] = {k: "" for k in SECTION_PATTERNS}
        sections["other"] = ""

        current = "other"
        for line in lines:
            matched_section = self._match_section(line)
            if matched_section:
                current = matched_section
                continue
            sections[current] += (" " + line).strip() + "\n"

        return {k: v.strip() for k, v in sections.items() if v.strip()}

    @staticmethod
    def _match_section(line: str) -> str | None:
        for section, pattern in SECTION_PATTERNS.items():
            if re.fullmatch(pattern, line):
                return section
        return None
