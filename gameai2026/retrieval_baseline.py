from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity

from .features import DEFAULT_NUMERIC_KEYS, FeatureBundle, build_feature_bundle
from .label_builder import (
    LABEL_SEGMENT_MIDDLE,
    LABEL_SEGMENT_PREFIX,
    LABEL_SEGMENT_TAIL,
    PLAN_EMPTY_TOKEN,
    build_training_label,
    extract_future_action_slots,
    future_action_text,
    infer_future_action_slots_from_label,
    split_label_segments,
)
from .local_structure_enhancer import (
    OTHER_STRUCTURE_LABEL,
    apply_local_structure_enhancement,
    build_action_tail_candidates,
    build_grenade_pair_candidates,
    collapse_action_tail_label,
    collapse_grenade_pair_label,
)
from .parser import MatchSample, iter_dataset_files, load_sample
from .text_generator import render_prediction


@dataclass(slots=True)
class SampleRecord:
    sample_path: str
    sample_type: str
    file_stem: str
    main_player_id: str | None
    decision_type: str
    summary_text: str
    numeric_features: dict[str, float]
    label_text: str = ""
    future_action_1: str = PLAN_EMPTY_TOKEN
    future_action_2: str = PLAN_EMPTY_TOKEN
    future_action_3: str = PLAN_EMPTY_TOKEN
    future_action_text: str = PLAN_EMPTY_TOKEN
    sample: MatchSample | None = None
    feature_bundle: FeatureBundle | None = None


@dataclass(slots=True)
class SlotPredictor:
    classifier: SGDClassifier | None = None
    constant_label: str | None = None


@dataclass(slots=True)
class RetrievalBucket:
    sample_type: str
    records: list[SampleRecord]
    vectorizer: TfidfVectorizer
    text_matrix: Any
    numeric_scaled: np.ndarray
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    model_matrix: Any
    first_action_predictor: SlotPredictor | None = None
    second_action_predictor: SlotPredictor | None = None
    action_tail_predictor: SlotPredictor | None = None
    grenade_pair_predictor: SlotPredictor | None = None
    action_tail_candidates: tuple[str, ...] = ()
    grenade_pair_candidates: tuple[str, ...] = ()


def normalize_plan_token(value: Any) -> str:
    if value is None:
        return PLAN_EMPTY_TOKEN
    if pd.isna(value):
        return PLAN_EMPTY_TOKEN
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return PLAN_EMPTY_TOKEN
    return text


def populate_future_action_fields(record: SampleRecord) -> SampleRecord:
    if (
        record.future_action_1 != PLAN_EMPTY_TOKEN
        or record.future_action_2 != PLAN_EMPTY_TOKEN
        or record.future_action_3 != PLAN_EMPTY_TOKEN
    ):
        if record.future_action_text == PLAN_EMPTY_TOKEN:
            tokens = [
                token
                for token in (record.future_action_1, record.future_action_2, record.future_action_3)
                if token != PLAN_EMPTY_TOKEN
            ]
            record.future_action_text = " -> ".join(tokens) if tokens else PLAN_EMPTY_TOKEN
        return record

    sample_path = Path(record.sample_path)
    if record.sample is not None or sample_path.exists():
        sample = record.sample if record.sample is not None else load_sample(sample_path)
        if record.sample is None:
            record.sample = sample

        slot_1, slot_2, slot_3 = extract_future_action_slots(sample, max_slots=3)
        record.future_action_1 = slot_1
        record.future_action_2 = slot_2
        record.future_action_3 = slot_3
        record.future_action_text = future_action_text(sample, max_slots=3)
        return record

    if record.label_text:
        slot_1, slot_2, slot_3 = infer_future_action_slots_from_label(record.label_text, max_slots=3)
        record.future_action_1 = slot_1
        record.future_action_2 = slot_2
        record.future_action_3 = slot_3
        tokens = [token for token in (slot_1, slot_2, slot_3) if token != PLAN_EMPTY_TOKEN]
        record.future_action_text = " -> ".join(tokens) if tokens else PLAN_EMPTY_TOKEN
    return record


def record_to_row(record: SampleRecord) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_path": record.sample_path,
        "sample_type": record.sample_type,
        "file_stem": record.file_stem,
        "main_player_id": record.main_player_id or "",
        "decision_type": record.decision_type,
        "summary_text": record.summary_text,
        "label_text": record.label_text,
        "future_action_1": record.future_action_1,
        "future_action_2": record.future_action_2,
        "future_action_3": record.future_action_3,
        "future_action_text": record.future_action_text,
    }
    for key in DEFAULT_NUMERIC_KEYS:
        row[key] = record.numeric_features.get(key, 0.0)
    return row


def records_to_dataframe(records: list[SampleRecord]) -> pd.DataFrame:
    return pd.DataFrame([record_to_row(record) for record in records])


def dataframe_to_records(df: pd.DataFrame) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for _, row in df.iterrows():
        numeric_features = {key: float(row.get(key, 0.0)) for key in DEFAULT_NUMERIC_KEYS}
        main_player_id = row.get("main_player_id", "")
        if pd.isna(main_player_id) or str(main_player_id).strip() in {"", "nan", "None"}:
            main_player_id = None
        else:
            main_player_id = str(main_player_id)
        records.append(
            SampleRecord(
                sample_path=str(row["sample_path"]),
                sample_type=str(row["sample_type"]),
                file_stem=str(row["file_stem"]),
                main_player_id=main_player_id,
                decision_type=str(row["decision_type"]),
                summary_text=str(row["summary_text"]),
                numeric_features=numeric_features,
                label_text=str(row.get("label_text", "")),
                future_action_1=normalize_plan_token(row.get("future_action_1")),
                future_action_2=normalize_plan_token(row.get("future_action_2")),
                future_action_3=normalize_plan_token(row.get("future_action_3")),
                future_action_text=normalize_plan_token(row.get("future_action_text")),
                sample=None,
                feature_bundle=None,
            )
        )
    return records


def build_sample_record(path: str | Path, with_label: bool = True) -> SampleRecord:
    sample = load_sample(path)
    feature_bundle = build_feature_bundle(sample)
    label_text = build_training_label(sample, feature_bundle) if with_label else ""
    future_1, future_2, future_3 = extract_future_action_slots(sample, max_slots=3) if with_label else (
        PLAN_EMPTY_TOKEN,
        PLAN_EMPTY_TOKEN,
        PLAN_EMPTY_TOKEN,
    )

    return SampleRecord(
        sample_path=str(Path(path)),
        sample_type=sample.sample_type,
        file_stem=sample.file_stem,
        main_player_id=sample.main_player_id,
        decision_type=sample.decision_type,
        summary_text=feature_bundle.summary_text,
        numeric_features=feature_bundle.numeric_features,
        label_text=label_text,
        future_action_1=future_1,
        future_action_2=future_2,
        future_action_3=future_3,
        future_action_text=future_action_text(sample, max_slots=3) if with_label else PLAN_EMPTY_TOKEN,
        sample=sample,
        feature_bundle=feature_bundle,
    )


def collect_records(dataset_root: str | Path, with_label: bool = True, limit: int | None = None) -> list[SampleRecord]:
    paths = iter_dataset_files(dataset_root)
    if limit is not None:
        paths = paths[:limit]
    return [build_sample_record(path, with_label=with_label) for path in paths]


def numeric_matrix_from_records(records: list[SampleRecord]) -> np.ndarray:
    return np.array(
        [
            [record.numeric_features.get(key, 0.0) for key in DEFAULT_NUMERIC_KEYS]
            for record in records
        ],
        dtype=np.float32,
    )


def fit_slot_predictor(model_matrix: Any, labels: list[str]) -> SlotPredictor | None:
    if not labels:
        return None

    unique_labels = sorted(set(labels))
    if len(unique_labels) == 1:
        return SlotPredictor(classifier=None, constant_label=unique_labels[0])

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=1500,
        tol=1e-3,
        random_state=42,
    )
    classifier.fit(model_matrix, np.array(labels, dtype=object))
    return SlotPredictor(classifier=classifier, constant_label=None)


def top_probabilities(prob_map: dict[str, float], top_n: int = 3) -> list[tuple[str, float]]:
    return sorted(prob_map.items(), key=lambda item: item[1], reverse=True)[:top_n]


def join_label_segments(first: str | None, second: str | None, third: str | None) -> str | None:
    if not first or not second or not third:
        return None
    return f"{LABEL_SEGMENT_PREFIX}{first}{LABEL_SEGMENT_MIDDLE}{second}{LABEL_SEGMENT_TAIL}{third}。"


def extract_summary_tokens(summary_text: str, prefix: str) -> list[str]:
    values: list[str] = []
    for token in summary_text.split():
        if token.startswith(prefix):
            values.append(token[len(prefix):])
    return values


def feature_bundle_from_record(record: SampleRecord) -> FeatureBundle:
    scope_tokens = extract_summary_tokens(record.summary_text, "镜态=")
    recent_actions = extract_summary_tokens(record.summary_text, "动作=")
    return FeatureBundle(
        numeric_features=record.numeric_features,
        summary_text=record.summary_text,
        latest_scope_state=scope_tokens[0] if scope_tokens else "未知",
        recent_actions=recent_actions[-4:],
        visible_enemy_count=int(record.numeric_features.get("visible_enemy_count", 0.0)),
        teammate_down_count=int(record.numeric_features.get("recent_teammate_down_count_5s", 0.0)),
    )


def sample_from_record(record: SampleRecord) -> MatchSample:
    return MatchSample(
        source_path=Path(record.sample_path),
        sample_type=record.sample_type,
        file_stem=record.file_stem,
        events=[],
        history_events=[],
        future_events=[],
        main_player_id=record.main_player_id,
        decision_type=record.decision_type,
        target_player_id=None,
    )


class RetrievalBaseline:
    def __init__(
        self,
        text_weight: float = 0.78,
        numeric_weight: float = 0.22,
        candidate_pool: int = 60,
        first_action_weight: float = 0.22,
        second_action_weight: float = 0.12,
        enable_action_tail_classifier: bool = False,
        enable_grenade_pair_classifier: bool = False,
    ) -> None:
        self.text_weight = text_weight
        self.numeric_weight = numeric_weight
        self.candidate_pool = candidate_pool
        self.first_action_weight = first_action_weight
        self.second_action_weight = second_action_weight
        self.enable_action_tail_classifier = enable_action_tail_classifier
        self.enable_grenade_pair_classifier = enable_grenade_pair_classifier
        self.buckets: dict[str, RetrievalBucket] = {}

    def fit(self, records: list[SampleRecord]) -> "RetrievalBaseline":
        grouped: dict[str, list[SampleRecord]] = {}
        for record in records:
            populate_future_action_fields(record)
            grouped.setdefault(record.sample_type, []).append(record)

        self.buckets.clear()
        for sample_type, bucket_records in grouped.items():
            vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
            text_matrix = vectorizer.fit_transform([record.summary_text for record in bucket_records])

            numeric = numeric_matrix_from_records(bucket_records)
            mean = numeric.mean(axis=0)
            std = numeric.std(axis=0)
            std[std < 1e-6] = 1.0
            scaled = (numeric - mean) / std
            model_matrix = hstack([text_matrix, csr_matrix(scaled)], format="csr")

            first_labels = [record.future_action_1 for record in bucket_records]
            second_labels = [record.future_action_2 for record in bucket_records]
            action_tail_candidates = build_action_tail_candidates([record.label_text for record in bucket_records])
            grenade_pair_candidates = build_grenade_pair_candidates([record.label_text for record in bucket_records])

            action_tail_labels = []
            if sample_type == "Action":
                action_tail_labels = [
                    collapse_action_tail_label(record.label_text, action_tail_candidates)
                    for record in bucket_records
                ]

            grenade_pair_labels = []
            if sample_type == "Grenade":
                grenade_pair_labels = [
                    collapse_grenade_pair_label(record.label_text, grenade_pair_candidates)
                    for record in bucket_records
                ]

            self.buckets[sample_type] = RetrievalBucket(
                sample_type=sample_type,
                records=bucket_records,
                vectorizer=vectorizer,
                text_matrix=text_matrix,
                numeric_scaled=scaled,
                numeric_mean=mean,
                numeric_std=std,
                model_matrix=model_matrix,
                first_action_predictor=fit_slot_predictor(model_matrix, first_labels),
                second_action_predictor=fit_slot_predictor(model_matrix, second_labels),
                action_tail_predictor=fit_slot_predictor(model_matrix, action_tail_labels) if action_tail_labels else None,
                grenade_pair_predictor=fit_slot_predictor(model_matrix, grenade_pair_labels) if grenade_pair_labels else None,
                action_tail_candidates=action_tail_candidates,
                grenade_pair_candidates=grenade_pair_candidates,
            )
        return self

    def _query_bucket(self, record: SampleRecord) -> RetrievalBucket:
        if record.sample_type in self.buckets:
            return self.buckets[record.sample_type]
        if not self.buckets:
            raise RuntimeError("RetrievalBaseline is not fitted.")
        return next(iter(self.buckets.values()))

    def _predict_slot_probabilities(self, predictor: SlotPredictor | None, query_matrix: Any) -> dict[str, float]:
        if predictor is None:
            return {}
        if predictor.constant_label is not None:
            return {predictor.constant_label: 1.0}
        if predictor.classifier is None:
            return {}

        probabilities = predictor.classifier.predict_proba(query_matrix)[0]
        return {
            str(label): float(score)
            for label, score in zip(predictor.classifier.classes_, probabilities)
        }

    def _future_action_bonus(
        self,
        neighbor: SampleRecord,
        first_probs: dict[str, float],
        second_probs: dict[str, float],
        sample_type: str,
    ) -> float:
        difficulty_multiplier = 1.15 if sample_type in {"Action", "SkillStart"} else 1.0
        empty_discount = 0.45

        first_bonus = first_probs.get(neighbor.future_action_1, 0.0)
        second_bonus = second_probs.get(neighbor.future_action_2, 0.0)

        if neighbor.future_action_1 == PLAN_EMPTY_TOKEN:
            first_bonus *= empty_discount
        if neighbor.future_action_2 == PLAN_EMPTY_TOKEN:
            second_bonus *= empty_discount

        return difficulty_multiplier * (
            self.first_action_weight * first_bonus
            + self.second_action_weight * second_bonus
        )

    def predict_one(self, record: SampleRecord, top_k: int = 5) -> dict[str, Any]:
        bucket = self._query_bucket(record)
        text_vector = bucket.vectorizer.transform([record.summary_text])
        text_scores = cosine_similarity(text_vector, bucket.text_matrix)[0]

        query_numeric = np.array(
            [[record.numeric_features.get(key, 0.0) for key in DEFAULT_NUMERIC_KEYS]],
            dtype=np.float32,
        )
        query_numeric = (query_numeric - bucket.numeric_mean) / bucket.numeric_std
        numeric_scores = cosine_similarity(query_numeric, bucket.numeric_scaled)[0]

        base_scores = self.text_weight * text_scores + self.numeric_weight * numeric_scores
        query_model_matrix = hstack([text_vector, csr_matrix(query_numeric)], format="csr")
        first_probs = self._predict_slot_probabilities(bucket.first_action_predictor, query_model_matrix)
        second_probs = self._predict_slot_probabilities(bucket.second_action_predictor, query_model_matrix)
        action_tail_probs = self._predict_slot_probabilities(bucket.action_tail_predictor, query_model_matrix)
        grenade_pair_probs = self._predict_slot_probabilities(bucket.grenade_pair_predictor, query_model_matrix)

        candidate_pool = min(len(bucket.records), max(top_k, self.candidate_pool))
        candidate_indices = np.argsort(base_scores)[::-1][:candidate_pool]

        reranked_items: list[tuple[SampleRecord, int, float]] = []
        for index in candidate_indices:
            neighbor = bucket.records[int(index)]
            rerank_score = float(base_scores[int(index)]) + self._future_action_bonus(
                neighbor,
                first_probs,
                second_probs,
                record.sample_type,
            )
            reranked_items.append((neighbor, int(index), rerank_score))

        reranked_items.sort(key=lambda item: item[2], reverse=True)
        selected_items = reranked_items[:top_k]

        retrieved_records = [item[0] for item in selected_items]
        retrieved_labels = [item.label_text for item in retrieved_records if item.label_text]
        retrieved_scores = [item[2] for item in selected_items[: len(retrieved_labels)]]

        sample_path = Path(record.sample_path)
        if record.sample is not None:
            sample = record.sample
        elif sample_path.exists():
            sample = load_sample(sample_path)
        else:
            sample = sample_from_record(record)

        if record.feature_bundle is not None:
            feature_bundle = record.feature_bundle
        elif record.sample is not None or sample_path.exists():
            feature_bundle = build_feature_bundle(sample)
        else:
            feature_bundle = feature_bundle_from_record(record)
        prediction_text, generation_strategy = render_prediction(
            sample,
            feature_bundle,
            retrieved_labels,
            retrieved_scores,
        )
        action_tail_input = action_tail_probs if self.enable_action_tail_classifier else None
        grenade_pair_input = grenade_pair_probs if self.enable_grenade_pair_classifier else None
        enhanced_text, enhancement_reason, enhancement_debug = apply_local_structure_enhancement(
            record.sample_type,
            prediction_text,
            feature_bundle,
            action_tail_probabilities=action_tail_input,
            grenade_pair_probabilities=grenade_pair_input,
        )
        if enhancement_reason:
            prediction_text = enhanced_text
            generation_strategy = f"{generation_strategy}+{enhancement_reason}"

        _, final_second_clause, final_third_clause = split_label_segments(prediction_text)

        top_first = top_probabilities(first_probs, top_n=3)
        top_second = top_probabilities(second_probs, top_n=3)
        top_action_tail = top_probabilities(action_tail_probs, top_n=3)
        top_grenade_pair = top_probabilities(grenade_pair_probs, top_n=3)

        return {
            "prediction_text": prediction_text,
            "generation_strategy": generation_strategy,
            "predicted_future_action_1": top_first[0][0] if top_first else PLAN_EMPTY_TOKEN,
            "predicted_future_action_2": top_second[0][0] if top_second else PLAN_EMPTY_TOKEN,
            "predicted_action_tail_clause": top_action_tail[0][0] if top_action_tail else OTHER_STRUCTURE_LABEL,
            "predicted_grenade_pair": top_grenade_pair[0][0] if top_grenade_pair else OTHER_STRUCTURE_LABEL,
            "predicted_second_clause": final_second_clause or "",
            "predicted_third_clause": final_third_clause or "",
            "local_enhancement_reason": enhancement_reason,
            "action_tail_candidates": list(bucket.action_tail_candidates),
            "grenade_pair_candidates": list(bucket.grenade_pair_candidates),
            **enhancement_debug,
            "neighbors": [
                {
                    "sample_path": neighbor.sample_path,
                    "label_text": neighbor.label_text,
                    "score": score,
                    "base_score": float(base_scores[index]),
                    "future_action_text": neighbor.future_action_text,
                    "future_action_1": neighbor.future_action_1,
                    "future_action_2": neighbor.future_action_2,
                }
                for neighbor, index, score in selected_items
            ],
        }
