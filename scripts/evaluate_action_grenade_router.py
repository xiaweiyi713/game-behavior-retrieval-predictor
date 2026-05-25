from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gameai2026.action_grenade_router import (
    FINAL_BASELINE_PARAMS,
    FINAL_TUNED_AG_PARAMS,
    ROUTED_SAMPLE_TYPES,
    filter_records,
    fit_action_grenade_router,
    make_baseline,
)
from gameai2026.retrieval_baseline import collect_records, dataframe_to_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Action/Grenade conditional router on a balanced holdout split.")
    parser.add_argument("--train-root", type=Path, default=ROOT / "final_all")
    parser.add_argument("--train-records", type=Path, default=ROOT / "outputs" / "train_records_full_with_future.csv")
    parser.add_argument("--eval-csv", type=Path, default=ROOT / "outputs" / "holdout_eval_action_grenade_router.csv")
    parser.add_argument("--router-oof-csv", type=Path, default=ROOT / "outputs" / "router_action_grenade_oof.csv")
    parser.add_argument("--holdout-per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--router-oof-folds", type=int, default=3)
    parser.add_argument("--enable-action-tail-cls", action="store_true")
    parser.add_argument("--enable-grenade-pair-cls", action="store_true")
    parser.add_argument("--rebuild-train-records", action="store_true")
    return parser.parse_args()


def load_or_build_records(args: argparse.Namespace):
    if args.train_records.exists() and not args.rebuild_train_records:
        df = pd.read_csv(args.train_records)
        return dataframe_to_records(df)
    return collect_records(args.train_root, with_label=True)


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
        for index, record in enumerate(bucket):
            if index in holdout_set:
                val_records.append(record)
            else:
                train_records.append(record)
    return train_records, val_records


def update_metric(metrics: dict, strategy_name: str, sample_type: str, exact: int) -> None:
    metrics[strategy_name]["overall_total"] += 1
    metrics[strategy_name]["overall_exact"] += exact
    metrics[strategy_name]["per_type"][sample_type]["total"] += 1
    metrics[strategy_name]["per_type"][sample_type]["exact"] += exact


def print_strategy_summary(metrics: dict, strategy_name: str) -> None:
    overall_total = metrics[strategy_name]["overall_total"]
    overall_exact = metrics[strategy_name]["overall_exact"] / overall_total if overall_total else 0.0
    print(f"{strategy_name}_overall_exact={overall_exact:.6f}")
    for sample_type in sorted(metrics[strategy_name]["per_type"]):
        info = metrics[strategy_name]["per_type"][sample_type]
        sample_exact = info["exact"] / info["total"] if info["total"] else 0.0
        print(f"  {strategy_name}:{sample_type} exact={sample_exact:.6f} total={info['total']}")


def main() -> None:
    args = parse_args()
    records = load_or_build_records(args)
    train_records, val_records = build_holdout_split(records, args.holdout_per_type, args.seed)

    baseline_params = dict(FINAL_BASELINE_PARAMS)
    tuned_params = dict(FINAL_TUNED_AG_PARAMS)
    for params in (baseline_params, tuned_params):
        params["enable_action_tail_classifier"] = args.enable_action_tail_cls
        params["enable_grenade_pair_classifier"] = args.enable_grenade_pair_cls

    stable_baseline = make_baseline(baseline_params).fit(train_records)

    routed_train_records = filter_records(train_records, ROUTED_SAMPLE_TYPES)
    routed_tuned_baseline = make_baseline(tuned_params).fit(routed_train_records)
    router, router_rows = fit_action_grenade_router(
        routed_train_records,
        top_k=args.top_k,
        oof_folds=args.router_oof_folds,
        seed=args.seed,
        route_types=ROUTED_SAMPLE_TYPES,
        base_params=baseline_params,
        tuned_params=tuned_params,
    )

    metrics = {
        "base": {
            "overall_total": 0,
            "overall_exact": 0,
            "per_type": defaultdict(lambda: {"total": 0, "exact": 0}),
        },
        "typed_mix": {
            "overall_total": 0,
            "overall_exact": 0,
            "per_type": defaultdict(lambda: {"total": 0, "exact": 0}),
        },
        "router_mix": {
            "overall_total": 0,
            "overall_exact": 0,
            "per_type": defaultdict(lambda: {"total": 0, "exact": 0}),
        },
    }

    rows: list[dict] = []
    for record in val_records:
        base_result = stable_baseline.predict_one(record, top_k=args.top_k)
        base_prediction = str(base_result["prediction_text"])
        base_exact = int(base_prediction == record.label_text)
        update_metric(metrics, "base", record.sample_type, base_exact)

        typed_prediction = base_prediction
        typed_exact = base_exact

        router_prediction = base_prediction
        router_exact = base_exact
        chosen_source = "base"
        router_reason = "stable_type"
        router_probability_tuned = 0.0
        router_threshold = 0.5

        tuned_result: dict[str, object] | None = None
        tuned_prediction = ""
        tuned_exact = base_exact

        if record.sample_type in ROUTED_SAMPLE_TYPES:
            tuned_result = routed_tuned_baseline.predict_one(record, top_k=args.top_k)
            tuned_prediction = str(tuned_result["prediction_text"])
            tuned_exact = int(tuned_prediction == record.label_text)

            typed_prediction = tuned_prediction
            typed_exact = tuned_exact

            route_info = router.route_prediction(record, base_result, tuned_result)
            chosen_result = route_info["chosen_result"]
            router_prediction = str(chosen_result["prediction_text"])
            router_exact = int(router_prediction == record.label_text)
            chosen_source = str(route_info["chosen_source"])
            router_reason = str(route_info["router_reason"])
            router_probability_tuned = float(route_info["router_probability_tuned"])
            router_threshold = float(route_info["router_threshold"])

        update_metric(metrics, "typed_mix", record.sample_type, typed_exact)
        update_metric(metrics, "router_mix", record.sample_type, router_exact)

        rows.append(
            {
                "sample_type": record.sample_type,
                "file_stem": record.file_stem,
                "reference_text": record.label_text,
                "base_prediction_text": base_prediction,
                "base_exact_match": base_exact,
                "typed_mix_prediction_text": typed_prediction,
                "typed_mix_exact_match": typed_exact,
                "tuned_prediction_text": tuned_prediction,
                "tuned_exact_match": tuned_exact,
                "router_prediction_text": router_prediction,
                "router_exact_match": router_exact,
                "router_chosen_source": chosen_source,
                "router_reason": router_reason,
                "router_probability_tuned": router_probability_tuned,
                "router_threshold": router_threshold,
            }
        )

    eval_df = pd.DataFrame(rows)
    args.eval_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(args.eval_csv, index=False, encoding="utf-8-sig")

    router_debug_df = pd.DataFrame(
        [
            {
                "sample_type": row.sample_type,
                "file_stem": row.file_stem,
                "base_prediction": row.base_prediction,
                "tuned_prediction": row.tuned_prediction,
                "base_exact": row.base_exact,
                "tuned_exact": row.tuned_exact,
                "preferred_source": row.preferred_source or "",
                "prediction_same": int(row.prediction_same),
                "base_top1_score": row.base_top1_score,
                "tuned_top1_score": row.tuned_top1_score,
            }
            for row in router_rows
        ]
    )
    router_debug_df.to_csv(args.router_oof_csv, index=False, encoding="utf-8-sig")

    print(f"validation_size={len(val_records)}")
    print(
        "local_structure_flags:"
        f" action_tail={int(args.enable_action_tail_cls)}"
        f" grenade_pair={int(args.enable_grenade_pair_cls)}"
    )
    print_strategy_summary(metrics, "base")
    print_strategy_summary(metrics, "typed_mix")
    print_strategy_summary(metrics, "router_mix")
    for sample_type in ROUTED_SAMPLE_TYPES:
        info = router.routers.get(sample_type)
        if info is not None:
            print(
                f"router_{sample_type}: default={info.default_source} "
                f"threshold={info.threshold:.2f} metrics={info.metrics}"
            )
    print(f"saved_eval={args.eval_csv}")
    print(f"saved_router_oof={args.router_oof_csv}")


if __name__ == "__main__":
    main()
