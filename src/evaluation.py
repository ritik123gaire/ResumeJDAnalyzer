from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence


@dataclass
class PairExample:
    resume_id: str
    jd_id: str
    resume_text: str
    jd_text: str
    label: float


def load_labeled_pairs(path: str) -> List[PairExample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = []
    for item in data:
        pairs.append(
            PairExample(
                resume_id=item["resume_id"],
                jd_id=item["jd_id"],
                resume_text=item["resume_text"],
                jd_text=item["jd_text"],
                label=float(item["label"]),
            )
        )
    return pairs


def ndcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    rel = list(relevances)[:k]
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(sorted(rel, reverse=True)))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(rank_positions: Sequence[int]) -> float:
    vals = [1.0 / p for p in rank_positions if p > 0]
    return sum(vals) / len(vals) if vals else 0.0


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def measure_latency(scorer: Callable[[str, str], float], pairs: Sequence[PairExample], repeats: int = 1) -> Dict[str, float]:
    start = time.perf_counter()
    count = 0
    for _ in range(repeats):
        for pair in pairs:
            scorer(pair.resume_text, pair.jd_text)
            count += 1
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / max(count, 1)) * 1000.0
    throughput = count / elapsed if elapsed else 0.0
    return {"avg_latency_ms": avg_ms, "throughput_pairs_per_sec": throughput}


def run_binary_metrics_from_scores(labels: Sequence[float], scores: Sequence[float], threshold: float = 3.5) -> Dict[str, float]:
    y_true = [1 if l >= threshold else 0 for l in labels]
    y_pred = [1 if s >= (threshold / 5.0) else 0 for s in scores]
    return precision_recall_f1(y_true, y_pred)
