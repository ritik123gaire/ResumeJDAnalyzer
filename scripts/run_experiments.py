from __future__ import annotations

import json
from pathlib import Path

from src.evaluation import load_labeled_pairs, measure_latency, run_binary_metrics_from_scores
from src.matching.biencoder_matcher import BiEncoderMatcher
from src.matching.crossencoder_matcher import CrossEncoderMatcher
from src.matching.tfidf_matcher import TFIDFMatcher


DATA_PATH = "data/evaluation/labeled_pairs.json"
OUT_PATH = "results/metrics/experiment_summary.json"


def main() -> None:
    pairs = load_labeled_pairs(DATA_PATH)

    tfidf = TFIDFMatcher()
    bi = BiEncoderMatcher()
    cross = CrossEncoderMatcher()

    labels = [p.label for p in pairs]

    tfidf_scores = [tfidf.score(p.resume_text, p.jd_text).score for p in pairs]
    bi_scores = [bi.score(p.resume_text, p.jd_text) for p in pairs]
    cross_scores = [cross.score(p.resume_text, p.jd_text) for p in pairs]

    results = {
        "tfidf": {
            "binary_metrics": run_binary_metrics_from_scores(labels, tfidf_scores),
            "latency": measure_latency(lambda a, b: tfidf.score(a, b).score, pairs, repeats=5),
        },
        "bi_encoder": {
            "binary_metrics": run_binary_metrics_from_scores(labels, bi_scores),
            "latency": measure_latency(bi.score, pairs, repeats=5),
        },
        "cross_encoder": {
            "binary_metrics": run_binary_metrics_from_scores(labels, cross_scores),
            "latency": measure_latency(cross.score, pairs, repeats=5),
        },
    }

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
