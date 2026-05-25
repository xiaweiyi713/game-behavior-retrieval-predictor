from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gameai2026.retrieval_baseline import collect_records, records_to_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training artifacts for the finals baseline.")
    parser.add_argument("--train-root", type=Path, default=ROOT / "final_all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "train_records.csv")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_records(args.train_root, with_label=True, limit=args.limit)
    df = records_to_dataframe(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"saved {len(df)} records to {args.output}")
    if not df.empty:
        print(df["sample_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
