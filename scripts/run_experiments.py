from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    load_labeled_pairs,
    mean_reciprocal_rank,
    measure_latency,
    ndcg_at_k,
    run_binary_metrics_from_scores,
)
from src.matching.biencoder_matcher import BiEncoderMatcher
from src.matching.crossencoder_matcher import CrossEncoderMatcher
from src.matching.tfidf_matcher import TFIDFMatcher


DATA_PATH = "data/evaluation/labeled_pairs.json"
OUT_PATH = "results/metrics/experiment_summary.json"


def ranking_metrics(
    resume_ids: list[str],
    labels: list[float],
    scores: list[float],
    k: int = 5,
) -> dict[str, float]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rid, label, score in zip(resume_ids, labels, scores):
        grouped[rid].append((label, score))

    ndcgs: list[float] = []
    rr_positions: list[int] = []
    for items in grouped.values():
        ranked = sorted(items, key=lambda x: x[1], reverse=True)
        ranked_labels = [pair[0] for pair in ranked]
        ndcgs.append(ndcg_at_k(ranked_labels, k=min(k, len(ranked_labels))))

        first_relevant = 0
        for idx, rel in enumerate(ranked_labels, start=1):
            if rel >= 3.5:
                first_relevant = idx
                break
        if first_relevant:
            rr_positions.append(first_relevant)

    return {
        "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "mrr": mean_reciprocal_rank(rr_positions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matcher experiments.")
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help="Path to labeled pairs JSON.",
    )
    parser.add_argument(
        "--out",
        default=OUT_PATH,
        help="Path to write experiment summary JSON.",
    )
    args = parser.parse_args()

    pairs = load_labeled_pairs(args.data)

    tfidf = TFIDFMatcher()
    bi = BiEncoderMatcher()
    cross = CrossEncoderMatcher()

    resume_ids = [p.resume_id for p in pairs]
    labels = [p.label for p in pairs]

    tfidf_scores = [tfidf.score(p.resume_text, p.jd_text).score for p in pairs]
    bi_scores = [bi.score(p.resume_text, p.jd_text) for p in pairs]
    cross_scores = [cross.score(p.resume_text, p.jd_text) for p in pairs]

    results = {
        "tfidf": {
            "binary_metrics": run_binary_metrics_from_scores(labels, tfidf_scores),
            "ranking_metrics": ranking_metrics(resume_ids, labels, tfidf_scores),
            "latency": measure_latency(lambda a, b: tfidf.score(a, b).score, pairs, repeats=5),
        },
        "bi_encoder": {
            "binary_metrics": run_binary_metrics_from_scores(labels, bi_scores),
            "ranking_metrics": ranking_metrics(resume_ids, labels, bi_scores),
            "latency": measure_latency(bi.score, pairs, repeats=5),
        },
        "cross_encoder": {
            "binary_metrics": run_binary_metrics_from_scores(labels, cross_scores),
            "ranking_metrics": ranking_metrics(resume_ids, labels, cross_scores),
            "latency": measure_latency(cross.score, pairs, repeats=5),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
