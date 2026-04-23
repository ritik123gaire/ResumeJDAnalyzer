from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.matching.crossencoder_matcher import CrossEncoderMatcher, CrossEncoderTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder on labeled resume-JD pairs.")
    parser.add_argument("--data", type=str, default="data/evaluation/labeled_pairs.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default="models/cross_encoder")
    args = parser.parse_args()

    pairs = json.loads(Path(args.data).read_text(encoding="utf-8"))

    config = CrossEncoderTrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output,
    )

    matcher = CrossEncoderMatcher(model_name_or_path=config.model_name)
    matcher.fine_tune(train_pairs=pairs, config=config)

    print(f"Saved fine-tuned model to {args.output}")


if __name__ == "__main__":
    main()
