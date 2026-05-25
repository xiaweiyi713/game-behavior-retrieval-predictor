from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gameai2026.retrieval_baseline import (
    RetrievalBaseline,
    collect_records,
    dataframe_to_records,
    populate_future_action_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the retrieval pipeline on a balanced holdout split.")
    parser.add_argument("--train-root", type=Path, default=ROOT / "final_all")
    parser.add_argument("--train-records", type=Path, default=ROOT / "outputs" / "train_records_full.csv")
    parser.add_argument("--eval-csv", type=Path, default=ROOT / "outputs" / "holdout_eval_v3.csv")
    parser.add_argument("--holdout-per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=60)
    parser.add_argument("--text-weight", type=float, default=0.78)
    parser.add_argument("--numeric-weight", type=float, default=0.22)
    parser.add_argument("--first-action-weight", type=float, default=0.22)
    parser.add_argument("--second-action-weight", type=float, default=0.12)
    parser.add_argument("--enable-action-tail-cls", action="store_true")
    parser.add_argument("--enable-grenade-pair-cls", action="store_true")
    parser.add_argument("--rebuild-train-records", action="store_true")
    return parser.parse_args()


def load_or_build_records(args: argparse.Namespace):
    if args.train_records.exists() and not args.rebuild_train_records:
        df = pd.read_csv(args.train_records)
        return dataframe_to_records(df)
    return collect_records(args.train_root, with_label=True)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = list(prediction.strip())
    ref = list(reference.strip())
    if not pred or not ref:
        return 0.0

    dp = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i in range(1, len(pred) + 1):
        for j in range(1, len(ref) + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_holdout_split(records, holdout_per_type: int, seed: int):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for record in records:
        grouped[record.sample_type].append(record)

    train_records = []
    val_records = []
    for sample_type, bucket in grouped.items():
        indices = list(range(len(bucket)))
        rng.shuffle(indices)
        holdout_count = min(holdout_per_type, len(bucket))
        holdout_set = set(indices[:holdout_count])
        for idx, record in enumerate(bucket):
            if idx in holdout_set:
                val_records.append(record)
            else:
                train_records.append(record)
    return train_records, val_records


def main() -> None:
    args = parse_args()
    records = load_or_build_records(args)

    for record in records:
        populate_future_action_fields(record)

    train_records, val_records = build_holdout_split(records, args.holdout_per_type, args.seed)
    baseline = RetrievalBaseline(
        text_weight=args.text_weight,
        numeric_weight=args.numeric_weight,
        candidate_pool=args.candidate_pool,
        first_action_weight=args.first_action_weight,
        second_action_weight=args.second_action_weight,
        enable_action_tail_classifier=args.enable_action_tail_cls,
        enable_grenade_pair_classifier=args.enable_grenade_pair_cls,
    ).fit(train_records)

    rows = []
    exact_hits = 0
    rouge_scores: list[float] = []
    per_type = defaultdict(lambda: {"total": 0, "exact": 0, "rouge": []})
    strategy_counter = Counter()

    for record in val_records:
        result = baseline.predict_one(record, top_k=args.top_k)
        prediction = result["prediction_text"]
        reference = record.label_text
        exact = int(prediction == reference)
        rouge = rouge_l_f1(prediction, reference)

        rows.append(
            {
                "sample_type": record.sample_type,
                "file_stem": record.file_stem,
                "prediction_text": prediction,
                "reference_text": reference,
                "exact_match": exact,
                "rouge_l_f1": rouge,
                "generation_strategy": result.get("generation_strategy", ""),
                "predicted_future_action_1": result.get("predicted_future_action_1", ""),
                "predicted_future_action_2": result.get("predicted_future_action_2", ""),
            }
        )

        exact_hits += exact
        rouge_scores.append(rouge)
        per_type[record.sample_type]["total"] += 1
        per_type[record.sample_type]["exact"] += exact
        per_type[record.sample_type]["rouge"].append(rouge)
        strategy_counter[result.get("generation_strategy", "")] += 1

    eval_df = pd.DataFrame(rows)
    args.eval_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(args.eval_csv, index=False, encoding="utf-8-sig")

    total = len(val_records)
    overall_exact = exact_hits / total if total else 0.0
    overall_rouge = sum(rouge_scores) / total if total else 0.0

    print(f"validation_size={total}")
    print(
        "local_structure_flags:"
        f" action_tail={int(args.enable_action_tail_cls)}"
        f" grenade_pair={int(args.enable_grenade_pair_cls)}"
    )
    print(f"overall_exact={overall_exact:.6f}")
    print(f"overall_rouge_l_f1={overall_rouge:.6f}")
    print("per_type:")
    for sample_type in sorted(per_type):
        info = per_type[sample_type]
        avg_rouge = sum(info["rouge"]) / info["total"] if info["total"] else 0.0
        print(
            f"  {sample_type}: exact={info['exact'] / info['total']:.6f} "
            f"rouge={avg_rouge:.6f} total={info['total']}"
        )
    print("strategy_counts:")
    for name, count in strategy_counter.items():
        print(f"  {name or '<empty>'}: {count}")
    print(f"saved_eval={args.eval_csv}")


if __name__ == "__main__":
    main()
