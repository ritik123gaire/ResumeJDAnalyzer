from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader


@dataclass
class CrossEncoderTrainingConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    num_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 50
    output_dir: str = "models/cross_encoder"


class CrossEncoderMatcher:
    def __init__(self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512):
        self.model = CrossEncoder(model_name_or_path, num_labels=1, max_length=max_length)

    @staticmethod
    def _logit_to_unit_interval(logit: float) -> float:
        """MS MARCO-style cross-encoders return a single logit, not a probability."""
        x = float(np.clip(logit, -80.0, 80.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def score(self, resume_text: str, jd_text: str) -> float:
        raw = self.model.predict([(resume_text, jd_text)], convert_to_numpy=True)
        return float(np.clip(self._logit_to_unit_interval(float(raw.reshape(-1)[0])), 0.0, 1.0))

    def score_skill_pairs(
        self,
        resume_skills: List[str],
        jd_skills: List[str],
        threshold: float = 0.5,
    ) -> List[Tuple[str, str, float]]:
        if not resume_skills or not jd_skills:
            return []

        candidates: List[Tuple[str, str]] = []
        index: List[Tuple[str, str]] = []
        for jd_skill in jd_skills:
            for resume_skill in resume_skills:
                candidates.append((resume_skill, jd_skill))
                index.append((jd_skill, resume_skill))

        scores = self.model.predict(candidates, convert_to_numpy=True)
        ranked = sorted(
            [
                (index[i][0], index[i][1], self._logit_to_unit_interval(float(scores[i])))
                for i in range(len(scores))
            ],
            key=lambda x: x[2],
            reverse=True,
        )
        return [item for item in ranked if item[2] >= threshold]

    @staticmethod
    def build_training_examples(pairs: Iterable[Dict[str, object]]) -> List[InputExample]:
        examples: List[InputExample] = []
        for item in pairs:
            resume_text = str(item["resume_text"])
            jd_text = str(item["jd_text"])
            raw_label = float(item["label"])
            label = max(0.0, min(1.0, raw_label / 5.0))
            examples.append(InputExample(texts=[resume_text, jd_text], label=label))
        return examples

    def fine_tune(self, train_pairs: List[Dict[str, object]], config: CrossEncoderTrainingConfig) -> None:
        train_examples = self.build_training_examples(train_pairs)
        dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)

        self.model.fit(
            train_dataloader=dataloader,
            epochs=config.num_epochs,
            warmup_steps=config.warmup_steps,
            output_path=config.output_dir,
            optimizer_params={"lr": config.learning_rate},
            use_amp=False,
        )

    def rank_candidates(self, resume_text: str, jd_texts: Dict[str, str]) -> List[Tuple[str, float]]:
        ids = list(jd_texts.keys())
        pairs = [(resume_text, jd_texts[jid]) for jid in ids]
        scores = self.model.predict(pairs, convert_to_numpy=True)
        ranked = sorted(zip(ids, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [
            (jid, float(np.clip(self._logit_to_unit_interval(float(score)), 0.0, 1.0)))
            for jid, score in ranked
        ]
