from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from .features import bucket_distance
from .label_builder import PLAN_EMPTY_TOKEN, split_label_segments
from .retrieval_baseline import RetrievalBaseline, SampleRecord


ROUTED_SAMPLE_TYPES = ("Action", "Grenade")
MIN_DECISIVE_EXAMPLES = 80
MIN_CLASS_EXAMPLES = 12

FINAL_BASELINE_PARAMS = {
    "candidate_pool": 5,
    "text_weight": 0.78,
    "numeric_weight": 0.22,
    "first_action_weight": 0.0,
    "second_action_weight": 0.0,
}

FINAL_TUNED_AG_PARAMS = {
    "candidate_pool": 20,
    "text_weight": 0.78,
    "numeric_weight": 0.22,
    "first_action_weight": 0.18,
    "second_action_weight": 0.12,
}


@dataclass(slots=True)
class RouterTrainingRow:
    sample_type: str
    file_stem: str
    features: dict[str, Any]
    base_prediction: str
    tuned_prediction: str
    base_exact: int
    tuned_exact: int
    preferred_source: str | None
    base_top1_score: float
    tuned_top1_score: float
    prediction_same: bool


@dataclass(slots=True)
class SingleTypeRouter:
    sample_type: str
    default_source: str = "tuned"
    threshold: float = 0.5
    vectorizer: DictVectorizer | None = None
    scaler: MaxAbsScaler | None = None
    classifier: LogisticRegression | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def predict_proba_tuned(self, features: dict[str, Any]) -> float:
        if self.vectorizer is None or self.scaler is None or self.classifier is None:
            return 1.0 if self.default_source == "tuned" else 0.0
        matrix = self.vectorizer.transform([features])
        matrix = self.scaler.transform(matrix)
        return float(self.classifier.predict_proba(matrix)[0][1])

    def choose_source(self, features: dict[str, Any]) -> tuple[str, float, str]:
        if bool(features.get("prediction_same")):
            return "base", 0.5, "same_prediction"
        if self.vectorizer is None or self.scaler is None or self.classifier is None:
            probability_tuned = 1.0 if self.default_source == "tuned" else 0.0
            return self.default_source, probability_tuned, "router_default"
        probability_tuned = self.predict_proba_tuned(features)
        chosen_source = "tuned" if probability_tuned >= self.threshold else "base"
        return chosen_source, probability_tuned, "router_model"


@dataclass(slots=True)
class ActionGrenadeRouter:
    routers: dict[str, SingleTypeRouter]

    def route_prediction(
        self,
        record: SampleRecord,
        base_result: dict[str, Any],
        tuned_result: dict[str, Any],
    ) -> dict[str, Any]:
        router = self.routers.get(record.sample_type)
        if router is None:
            return {
                "chosen_result": base_result,
                "chosen_source": "base",
                "router_probability_tuned": 0.0,
                "router_threshold": 0.5,
                "router_reason": "no_router",
                "prediction_same": True,
            }

        features = build_router_features(record, base_result, tuned_result)
        chosen_source, probability_tuned, reason = router.choose_source(features)
        chosen_result = tuned_result if chosen_source == "tuned" else base_result
        return {
            "chosen_result": chosen_result,
            "chosen_source": chosen_source,
            "router_probability_tuned": probability_tuned,
            "router_threshold": router.threshold,
            "router_reason": reason,
            "prediction_same": bool(features.get("prediction_same")),
        }


def make_baseline(params: dict[str, float]) -> RetrievalBaseline:
    return RetrievalBaseline(
        text_weight=float(params["text_weight"]),
        numeric_weight=float(params["numeric_weight"]),
        candidate_pool=int(params["candidate_pool"]),
        first_action_weight=float(params["first_action_weight"]),
        second_action_weight=float(params["second_action_weight"]),
        enable_action_tail_classifier=bool(params.get("enable_action_tail_classifier", False)),
        enable_grenade_pair_classifier=bool(params.get("enable_grenade_pair_classifier", False)),
    )


def filter_records(records: list[SampleRecord], sample_types: tuple[str, ...] | list[str]) -> list[SampleRecord]:
    sample_type_set = set(sample_types)
    return [record for record in records if record.sample_type in sample_type_set]


def extract_summary_tokens(summary_text: str, prefix: str) -> list[str]:
    values: list[str] = []
    for token in summary_text.split():
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
    return values


def extract_recent_tokens(summary_text: str, prefix: str, max_items: int) -> list[str]:
    values = extract_summary_tokens(summary_text, prefix)
    values = list(reversed(values[-max_items:]))
    while len(values) < max_items:
        values.append(PLAN_EMPTY_TOKEN)
    return values


def clip_distance(value: float) -> float:
    if value >= 9999.0:
        return 250.0
    return min(float(value), 250.0)


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    prediction_text = str(result.get("prediction_text", ""))
    _, second_clause, third_clause = split_label_segments(prediction_text)

    neighbors = list(result.get("neighbors", []))
    top1 = neighbors[0] if len(neighbors) >= 1 else {}
    top2 = neighbors[1] if len(neighbors) >= 2 else {}

    top1_score = float(top1.get("score", 0.0)) if top1 else 0.0
    top2_score = float(top2.get("score", 0.0)) if top2 else 0.0

    return {
        "prediction_text": prediction_text,
        "generation_strategy": str(result.get("generation_strategy", "")),
        "predicted_future_action_1": str(result.get("predicted_future_action_1", PLAN_EMPTY_TOKEN)),
        "predicted_future_action_2": str(result.get("predicted_future_action_2", PLAN_EMPTY_TOKEN)),
        "second_clause": second_clause or PLAN_EMPTY_TOKEN,
        "third_clause": third_clause or PLAN_EMPTY_TOKEN,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "top1_gap": top1_score - top2_score,
        "top1_neighbor_future_action_1": str(top1.get("future_action_1", PLAN_EMPTY_TOKEN)),
        "top1_neighbor_future_action_2": str(top1.get("future_action_2", PLAN_EMPTY_TOKEN)),
    }


def build_router_features(
    record: SampleRecord,
    base_result: dict[str, Any],
    tuned_result: dict[str, Any],
) -> dict[str, Any]:
    base_summary = summarize_result(base_result)
    tuned_summary = summarize_result(tuned_result)

    numeric = record.numeric_features
    raw_enemy_distance = float(numeric.get("nearest_enemy_distance", 9999.0))
    raw_teammate_distance = float(numeric.get("nearest_teammate_distance", 9999.0))

    latest_scope_values = extract_summary_tokens(record.summary_text, "镜态=")
    latest_scope_state = latest_scope_values[0] if latest_scope_values else "未知"

    recent_actions = extract_recent_tokens(record.summary_text, "动作=", max_items=4)
    recent_skills = extract_recent_tokens(record.summary_text, "技能=", max_items=2)

    base_pair = f"{base_summary['predicted_future_action_1']}->{base_summary['predicted_future_action_2']}"
    tuned_pair = f"{tuned_summary['predicted_future_action_1']}->{tuned_summary['predicted_future_action_2']}"

    features: dict[str, Any] = {
        "sample_type": record.sample_type,
        "decision_type": record.decision_type or PLAN_EMPTY_TOKEN,
        "latest_scope_state": latest_scope_state,
        "nearest_enemy_bucket": bucket_distance(None if raw_enemy_distance >= 9999.0 else raw_enemy_distance),
        "nearest_teammate_bucket": bucket_distance(None if raw_teammate_distance >= 9999.0 else raw_teammate_distance),
        "visible_enemy_count": float(numeric.get("visible_enemy_count", 0.0)),
        "nearest_enemy_distance_clipped": clip_distance(raw_enemy_distance),
        "nearest_teammate_distance_clipped": clip_distance(raw_teammate_distance),
        "nearest_enemy_missing": float(raw_enemy_distance >= 9999.0),
        "nearest_teammate_missing": float(raw_teammate_distance >= 9999.0),
        "scope_open_ratio_5s": float(numeric.get("scope_open_ratio_5s", 0.0)),
        "avg_speed_3s": float(numeric.get("avg_speed_3s", 0.0)),
        "latest_speed": float(numeric.get("latest_speed", 0.0)),
        "recent_action_count_3s": float(numeric.get("recent_action_count_3s", 0.0)),
        "recent_skill_count_5s": float(numeric.get("recent_skill_count_5s", 0.0)),
        "recent_outgoing_hit_count_3s": float(numeric.get("recent_outgoing_hit_count_3s", 0.0)),
        "recent_incoming_hit_count_3s": float(numeric.get("recent_incoming_hit_count_3s", 0.0)),
        "recent_outgoing_hit_count_5s": float(numeric.get("recent_outgoing_hit_count_5s", 0.0)),
        "recent_incoming_hit_count_5s": float(numeric.get("recent_incoming_hit_count_5s", 0.0)),
        "recent_teammate_down_count_5s": float(numeric.get("recent_teammate_down_count_5s", 0.0)),
        "nearby_loot_count": float(numeric.get("nearby_loot_count", 0.0)),
        "nearby_high_loot_count": float(numeric.get("nearby_high_loot_count", 0.0)),
        "prediction_same": float(base_summary["prediction_text"] == tuned_summary["prediction_text"]),
        "generation_strategy_same": float(base_summary["generation_strategy"] == tuned_summary["generation_strategy"]),
        "base_generation_strategy": base_summary["generation_strategy"] or PLAN_EMPTY_TOKEN,
        "tuned_generation_strategy": tuned_summary["generation_strategy"] or PLAN_EMPTY_TOKEN,
        "base_top1_score": base_summary["top1_score"],
        "tuned_top1_score": tuned_summary["top1_score"],
        "base_top2_score": base_summary["top2_score"],
        "tuned_top2_score": tuned_summary["top2_score"],
        "base_top1_gap": base_summary["top1_gap"],
        "tuned_top1_gap": tuned_summary["top1_gap"],
        "top1_score_delta": tuned_summary["top1_score"] - base_summary["top1_score"],
        "top1_gap_delta": tuned_summary["top1_gap"] - base_summary["top1_gap"],
        "base_predicted_future_action_1": base_summary["predicted_future_action_1"],
        "base_predicted_future_action_2": base_summary["predicted_future_action_2"],
        "tuned_predicted_future_action_1": tuned_summary["predicted_future_action_1"],
        "tuned_predicted_future_action_2": tuned_summary["predicted_future_action_2"],
        "base_predicted_pair": base_pair,
        "tuned_predicted_pair": tuned_pair,
        "predicted_future_action_1_same": float(
            base_summary["predicted_future_action_1"] == tuned_summary["predicted_future_action_1"]
        ),
        "predicted_future_action_2_same": float(
            base_summary["predicted_future_action_2"] == tuned_summary["predicted_future_action_2"]
        ),
        "base_top1_neighbor_future_action_1": base_summary["top1_neighbor_future_action_1"],
        "base_top1_neighbor_future_action_2": base_summary["top1_neighbor_future_action_2"],
        "tuned_top1_neighbor_future_action_1": tuned_summary["top1_neighbor_future_action_1"],
        "tuned_top1_neighbor_future_action_2": tuned_summary["top1_neighbor_future_action_2"],
        "base_second_clause": base_summary["second_clause"],
        "base_third_clause": base_summary["third_clause"],
        "tuned_second_clause": tuned_summary["second_clause"],
        "tuned_third_clause": tuned_summary["third_clause"],
        "second_clause_same": float(base_summary["second_clause"] == tuned_summary["second_clause"]),
        "third_clause_same": float(base_summary["third_clause"] == tuned_summary["third_clause"]),
    }

    for index, action in enumerate(recent_actions, start=1):
        features[f"recent_action_{index}"] = action
    for index, skill in enumerate(recent_skills, start=1):
        features[f"recent_skill_{index}"] = skill

    return features


def build_router_training_row(
    record: SampleRecord,
    base_result: dict[str, Any],
    tuned_result: dict[str, Any],
) -> RouterTrainingRow:
    base_prediction = str(base_result.get("prediction_text", ""))
    tuned_prediction = str(tuned_result.get("prediction_text", ""))
    base_exact = int(base_prediction == record.label_text)
    tuned_exact = int(tuned_prediction == record.label_text)

    preferred_source: str | None = None
    if tuned_exact > base_exact:
        preferred_source = "tuned"
    elif base_exact > tuned_exact:
        preferred_source = "base"

    features = build_router_features(record, base_result, tuned_result)

    return RouterTrainingRow(
        sample_type=record.sample_type,
        file_stem=record.file_stem,
        features=features,
        base_prediction=base_prediction,
        tuned_prediction=tuned_prediction,
        base_exact=base_exact,
        tuned_exact=tuned_exact,
        preferred_source=preferred_source,
        base_top1_score=float(summarize_result(base_result)["top1_score"]),
        tuned_top1_score=float(summarize_result(tuned_result)["top1_score"]),
        prediction_same=base_prediction == tuned_prediction,
    )


def build_round_robin_folds(records: list[SampleRecord], n_splits: int, seed: int) -> list[list[SampleRecord]]:
    folds = [[] for _ in range(max(2, n_splits))]
    rng = Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    for index, record in enumerate(shuffled):
        folds[index % len(folds)].append(record)
    return [fold for fold in folds if fold]


def score_router_threshold(
    rows: list[RouterTrainingRow],
    probability_lookup: dict[str, float],
    threshold: float,
    default_source: str,
) -> float:
    hits = 0
    for row in rows:
        if row.prediction_same:
            chosen_source = "base"
        else:
            probability_tuned = probability_lookup.get(row.file_stem, 1.0 if default_source == "tuned" else 0.0)
            chosen_source = "tuned" if probability_tuned >= threshold else "base"
        hits += row.tuned_exact if chosen_source == "tuned" else row.base_exact
    return hits / len(rows) if rows else 0.0


def fit_single_type_router(
    sample_type: str,
    records: list[SampleRecord],
    top_k: int,
    oof_folds: int,
    seed: int,
    base_params: dict[str, float] | None = None,
    tuned_params: dict[str, float] | None = None,
) -> tuple[SingleTypeRouter, list[RouterTrainingRow]]:
    type_records = [record for record in records if record.sample_type == sample_type]
    if not type_records:
        router = SingleTypeRouter(sample_type=sample_type, default_source="base")
        router.metrics = {"oof_size": 0.0}
        return router, []

    base_params = base_params or FINAL_BASELINE_PARAMS
    tuned_params = tuned_params or FINAL_TUNED_AG_PARAMS

    folds = build_round_robin_folds(type_records, n_splits=min(max(2, oof_folds), len(type_records)), seed=seed)
    rows: list[RouterTrainingRow] = []

    for fold_index, val_records in enumerate(folds):
        train_records: list[SampleRecord] = []
        for other_index, fold_records in enumerate(folds):
            if other_index != fold_index:
                train_records.extend(fold_records)

        base_baseline = make_baseline(base_params).fit(train_records)
        tuned_baseline = make_baseline(tuned_params).fit(train_records)

        for record in val_records:
            base_result = base_baseline.predict_one(record, top_k=top_k)
            tuned_result = tuned_baseline.predict_one(record, top_k=top_k)
            rows.append(build_router_training_row(record, base_result, tuned_result))

    base_hits = sum(row.base_exact for row in rows)
    tuned_hits = sum(row.tuned_exact for row in rows)
    default_source = "tuned" if tuned_hits >= base_hits else "base"

    decisive_rows = [row for row in rows if row.preferred_source is not None and not row.prediction_same]
    label_values = {row.preferred_source for row in decisive_rows}
    tuned_examples = sum(1 for row in decisive_rows if row.preferred_source == "tuned")
    base_examples = sum(1 for row in decisive_rows if row.preferred_source == "base")

    router = SingleTypeRouter(sample_type=sample_type, default_source=default_source)

    if (
        len(label_values) >= 2
        and len(decisive_rows) >= MIN_DECISIVE_EXAMPLES
        and tuned_examples >= MIN_CLASS_EXAMPLES
        and base_examples >= MIN_CLASS_EXAMPLES
    ):
        vectorizer = DictVectorizer(sparse=True, sort=True)
        train_matrix = vectorizer.fit_transform([row.features for row in decisive_rows])
        scaler = MaxAbsScaler()
        train_matrix = scaler.fit_transform(train_matrix)

        labels = np.array([1 if row.preferred_source == "tuned" else 0 for row in decisive_rows], dtype=np.int8)
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        classifier.fit(train_matrix, labels)

        probability_lookup: dict[str, float] = {}
        for row in rows:
            if row.prediction_same:
                continue
            matrix = vectorizer.transform([row.features])
            matrix = scaler.transform(matrix)
            probability_lookup[row.file_stem] = float(classifier.predict_proba(matrix)[0][1])

        best_threshold = 0.5
        best_score = -1.0
        for threshold in np.linspace(0.30, 0.70, 21):
            score = score_router_threshold(rows, probability_lookup, float(threshold), default_source=default_source)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        router.vectorizer = vectorizer
        router.scaler = scaler
        router.classifier = classifier
        router.threshold = best_threshold
    else:
        best_score = score_router_threshold(rows, {}, threshold=0.5, default_source=default_source)

    router.metrics = {
        "oof_size": float(len(rows)),
        "oof_base_exact": base_hits / len(rows) if rows else 0.0,
        "oof_tuned_exact": tuned_hits / len(rows) if rows else 0.0,
        "oof_router_exact": best_score,
        "decisive_examples": float(len(decisive_rows)),
        "decisive_tuned_examples": float(tuned_examples),
        "decisive_base_examples": float(base_examples),
        "same_prediction_examples": float(sum(1 for row in rows if row.prediction_same)),
    }
    return router, rows


def fit_action_grenade_router(
    records: list[SampleRecord],
    top_k: int = 5,
    oof_folds: int = 3,
    seed: int = 42,
    route_types: tuple[str, ...] = ROUTED_SAMPLE_TYPES,
    base_params: dict[str, float] | None = None,
    tuned_params: dict[str, float] | None = None,
) -> tuple[ActionGrenadeRouter, list[RouterTrainingRow]]:
    routers: dict[str, SingleTypeRouter] = {}
    all_rows: list[RouterTrainingRow] = []

    for sample_type in route_types:
        router, rows = fit_single_type_router(
            sample_type=sample_type,
            records=records,
            top_k=top_k,
            oof_folds=oof_folds,
            seed=seed,
            base_params=base_params,
            tuned_params=tuned_params,
        )
        routers[sample_type] = router
        all_rows.extend(rows)

    return ActionGrenadeRouter(routers=routers), all_rows


def rows_to_debug_dicts(rows: list[RouterTrainingRow]) -> list[dict[str, Any]]:
    debug_rows: list[dict[str, Any]] = []
    for row in rows:
        item = {
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
        item.update(row.features)
        debug_rows.append(item)
    return debug_rows
