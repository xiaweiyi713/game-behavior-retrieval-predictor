from __future__ import annotations

from collections import Counter
from typing import Any

from .features import FeatureBundle
from .label_builder import (
    LABEL_SEGMENT_MIDDLE,
    LABEL_SEGMENT_PREFIX,
    LABEL_SEGMENT_TAIL,
    PLAN_EMPTY_TOKEN,
    split_label_segments,
)


OTHER_STRUCTURE_LABEL = "<OTHER>"

ACTION_TAIL_TOP_N = 10
ACTION_TAIL_MIN_COUNT = 5
GRENADE_PAIR_TOP_N = 15
GRENADE_PAIR_MIN_COUNT = 3

ACTION_TAIL_BASE_THRESHOLD = 0.34
ACTION_TAIL_MIN_GAP = 0.04
GRENADE_PAIR_BASE_THRESHOLD = 0.24
GRENADE_PAIR_MIN_GAP = 0.03

SCOPE_TAIL_CLAUSES = {
    "短暂收镜重新调整",
    "向左探头试探信息",
    "向右探头试探信息",
    "回正身位继续观察",
}
MOVEMENT_TAIL_CLAUSES = {
    "通过跳跃继续调整身位",
    "下蹲贴掩体稳住枪线",
    "重新起身寻找角度",
}


def rebuild_label(first: str, second: str, third: str) -> str:
    return f"{LABEL_SEGMENT_PREFIX}{first}{LABEL_SEGMENT_MIDDLE}{second}{LABEL_SEGMENT_TAIL}{third}。"


def build_action_tail_candidates(labels: list[str]) -> tuple[str, ...]:
    counter: Counter[str] = Counter()
    for label in labels:
        _, _, third = split_label_segments(label)
        if third:
            counter[third] += 1
    candidates = [
        clause
        for clause, count in counter.most_common(ACTION_TAIL_TOP_N)
        if count >= ACTION_TAIL_MIN_COUNT
    ]
    return tuple(candidates)


def build_grenade_pair_candidates(labels: list[str]) -> tuple[str, ...]:
    counter: Counter[str] = Counter()
    for label in labels:
        _, second, third = split_label_segments(label)
        if second and third:
            counter[f"{second} || {third}"] += 1
    candidates = [
        pair
        for pair, count in counter.most_common(GRENADE_PAIR_TOP_N)
        if count >= GRENADE_PAIR_MIN_COUNT
    ]
    return tuple(candidates)


def collapse_action_tail_label(label: str, candidates: tuple[str, ...]) -> str:
    _, _, third = split_label_segments(label)
    if not third:
        return OTHER_STRUCTURE_LABEL
    return third if third in set(candidates) else OTHER_STRUCTURE_LABEL


def collapse_grenade_pair_label(label: str, candidates: tuple[str, ...]) -> str:
    _, second, third = split_label_segments(label)
    if not second or not third:
        return OTHER_STRUCTURE_LABEL
    pair = f"{second} || {third}"
    return pair if pair in set(candidates) else OTHER_STRUCTURE_LABEL


def best_non_other_probability(
    prob_map: dict[str, float],
    other_label: str = OTHER_STRUCTURE_LABEL,
) -> tuple[str | None, float, float]:
    filtered = [(label, float(score)) for label, score in prob_map.items() if label != other_label]
    if not filtered:
        return None, 0.0, 0.0
    filtered.sort(key=lambda item: item[1], reverse=True)
    top_label, top_score = filtered[0]
    second_score = filtered[1][1] if len(filtered) > 1 else 0.0
    return top_label, top_score, top_score - second_score


def action_tail_threshold(
    predicted_clause: str,
    second_clause: str,
    feature_bundle: FeatureBundle,
) -> float:
    threshold = ACTION_TAIL_BASE_THRESHOLD
    numeric = feature_bundle.numeric_features
    recent_skill_count = float(numeric.get("recent_skill_count_5s", 0.0))
    latest_speed = float(numeric.get("latest_speed", 0.0))

    if second_clause == "继续保持开镜观察":
        threshold -= 0.03
    if predicted_clause in SCOPE_TAIL_CLAUSES and feature_bundle.latest_scope_state == "开镜":
        threshold -= 0.03
    if predicted_clause in {"向左探头试探信息", "向右探头试探信息"} and feature_bundle.visible_enemy_count >= 1:
        threshold -= 0.02
    if predicted_clause == "短暂收镜重新调整" and feature_bundle.latest_scope_state == "开镜":
        threshold -= 0.02
    if predicted_clause == "回正身位继续观察" and feature_bundle.latest_scope_state == "开镜":
        threshold -= 0.01
    if predicted_clause in MOVEMENT_TAIL_CLAUSES and latest_speed >= 80:
        threshold -= 0.02
    if predicted_clause == "利用技能效果继续博弈" and recent_skill_count > 0:
        threshold -= 0.03

    return max(0.18, threshold)


def grenade_pair_threshold(predicted_pair: str, feature_bundle: FeatureBundle) -> float:
    threshold = GRENADE_PAIR_BASE_THRESHOLD
    numeric = feature_bundle.numeric_features
    recent_skill_count = float(numeric.get("recent_skill_count_5s", 0.0))
    latest_speed = float(numeric.get("latest_speed", 0.0))

    if "观察爆炸结果并衔接走位" in predicted_pair:
        threshold -= 0.02
    if "利用技能效果继续博弈" in predicted_pair and recent_skill_count > 0:
        threshold -= 0.02
    if "通过跳跃继续调整身位" in predicted_pair and latest_speed >= 80:
        threshold -= 0.01
    if "短暂收镜重新调整 || 回正身位继续观察" in predicted_pair and feature_bundle.latest_scope_state == "开镜":
        threshold -= 0.02
    if "下蹲贴掩体稳住枪线 || 重新起身寻找角度" in predicted_pair and feature_bundle.visible_enemy_count >= 1:
        threshold -= 0.02

    return max(0.14, threshold)


def apply_action_tail_enhancement(
    prediction_text: str,
    feature_bundle: FeatureBundle,
    probability_map: dict[str, float],
) -> tuple[str, str, dict[str, Any]]:
    first, second, third = split_label_segments(prediction_text)
    debug = {
        "predicted_action_tail_clause": "",
        "predicted_action_tail_prob": 0.0,
        "predicted_action_tail_gap": 0.0,
        "action_tail_override_applied": 0,
    }
    if not first or not second or not third:
        return prediction_text, "", debug

    best_clause, best_prob, prob_gap = best_non_other_probability(probability_map)
    debug["predicted_action_tail_clause"] = best_clause or ""
    debug["predicted_action_tail_prob"] = best_prob
    debug["predicted_action_tail_gap"] = prob_gap
    if not best_clause:
        return prediction_text, "", debug

    threshold = action_tail_threshold(best_clause, second, feature_bundle)
    if best_prob < threshold or prob_gap < ACTION_TAIL_MIN_GAP or best_clause == third:
        return prediction_text, "", debug

    debug["action_tail_override_applied"] = 1
    return rebuild_label(first, second, best_clause), "action_tail_cls", debug


def apply_grenade_pair_enhancement(
    prediction_text: str,
    feature_bundle: FeatureBundle,
    probability_map: dict[str, float],
) -> tuple[str, str, dict[str, Any]]:
    first, second, third = split_label_segments(prediction_text)
    debug = {
        "predicted_grenade_pair": "",
        "predicted_grenade_pair_prob": 0.0,
        "predicted_grenade_pair_gap": 0.0,
        "grenade_pair_override_applied": 0,
    }
    if not first or not second or not third:
        return prediction_text, "", debug

    best_pair, best_prob, prob_gap = best_non_other_probability(probability_map)
    debug["predicted_grenade_pair"] = best_pair or ""
    debug["predicted_grenade_pair_prob"] = best_prob
    debug["predicted_grenade_pair_gap"] = prob_gap
    if not best_pair:
        return prediction_text, "", debug

    threshold = grenade_pair_threshold(best_pair, feature_bundle)
    if best_prob < threshold or prob_gap < GRENADE_PAIR_MIN_GAP:
        return prediction_text, "", debug

    pair_second, pair_third = [part.strip() for part in best_pair.split("||", 1)]
    if pair_second == second and pair_third == third:
        return prediction_text, "", debug

    debug["grenade_pair_override_applied"] = 1
    return rebuild_label(first, pair_second, pair_third), "grenade_pair_cls", debug


def apply_local_structure_enhancement(
    sample_type: str,
    prediction_text: str,
    feature_bundle: FeatureBundle,
    action_tail_probabilities: dict[str, float] | None = None,
    grenade_pair_probabilities: dict[str, float] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    debug: dict[str, Any] = {
        "predicted_action_tail_clause": "",
        "predicted_action_tail_prob": 0.0,
        "predicted_action_tail_gap": 0.0,
        "action_tail_override_applied": 0,
        "predicted_grenade_pair": "",
        "predicted_grenade_pair_prob": 0.0,
        "predicted_grenade_pair_gap": 0.0,
        "grenade_pair_override_applied": 0,
    }

    if sample_type == "Action" and action_tail_probabilities:
        new_text, reason, action_debug = apply_action_tail_enhancement(
            prediction_text,
            feature_bundle,
            action_tail_probabilities,
        )
        debug.update(action_debug)
        return new_text, reason, debug

    if sample_type == "Grenade" and grenade_pair_probabilities:
        new_text, reason, grenade_debug = apply_grenade_pair_enhancement(
            prediction_text,
            feature_bundle,
            grenade_pair_probabilities,
        )
        debug.update(grenade_debug)
        return new_text, reason, debug

    return prediction_text, "", debug
