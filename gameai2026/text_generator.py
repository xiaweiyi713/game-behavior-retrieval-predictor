from __future__ import annotations

from collections import defaultdict

from .features import FeatureBundle
from .label_builder import build_training_label
from .parser import MatchSample


TOP1_DIRECT_TYPES = {"Action", "BeingResuce", "Fire", "Grenade", "Looting", "SkillStart"}
TOP1_DIRECT_THRESHOLDS = {
    "Action": 0.90,
    "Grenade": 0.88,
}


def split_label_segments(label: str) -> tuple[str | None, str | None, str | None]:
    prefix = "主玩家先"
    middle = "，随后"
    tail = "，最后"
    if not label.startswith(prefix) or middle not in label or tail not in label:
        return None, None, None
    body = label.removeprefix(prefix).rstrip("。")
    first, rest = body.split(middle, 1)
    second, third = rest.split(tail, 1)
    return first.strip(), second.strip(), third.strip()


def aggregate_scores(values: list[tuple[str, float]]) -> dict[str, float]:
    counter: dict[str, float] = defaultdict(float)
    for text, score in values:
        if text:
            counter[text] += float(score)
    return counter


def has_recent_action(feature_bundle: FeatureBundle, keyword: str) -> bool:
    return any(keyword in action for action in feature_bundle.recent_actions)


def clause_context_bonus(
    clause: str,
    sample_type: str,
    segment_index: int,
    feature_bundle: FeatureBundle,
) -> float:
    numeric = feature_bundle.numeric_features
    visible_enemy_count = feature_bundle.visible_enemy_count
    latest_scope_state = feature_bundle.latest_scope_state
    recent_skill_count = numeric.get("recent_skill_count_5s", 0.0)
    outgoing_count = numeric.get("recent_outgoing_hit_count_3s", 0.0)
    incoming_count = numeric.get("recent_incoming_hit_count_3s", 0.0)
    latest_speed = numeric.get("latest_speed", 0.0)

    bonus = 0.0

    if "利用技能效果" in clause:
        bonus += 0.08 if recent_skill_count > 0 else -0.10
    if "根据技能效果重新调整站位" in clause:
        bonus += 0.12 if sample_type == "SkillStart" or recent_skill_count > 0 else -0.12
    if "顺势持续输出伤害" in clause:
        bonus += 0.08 if outgoing_count > 0 else -0.12
    if "同时注意规避来袭火力" in clause:
        bonus += 0.08 if incoming_count > 0 else -0.10
    if "保持对多名敌人的信息压制" in clause:
        bonus += 0.08 if visible_enemy_count >= 2 else -0.12
    if "继续保持开镜观察" in clause:
        bonus += 0.12 if latest_scope_state == "开镜" else -0.02
    if "短暂收镜重新调整" in clause:
        bonus += 0.06 if latest_scope_state == "开镜" else 0.02
    if "回正身位继续观察" in clause and latest_scope_state == "开镜":
        bonus += 0.05
    if "下蹲贴掩体稳住枪线" in clause and visible_enemy_count >= 1:
        bonus += 0.05
    if "重新起身寻找角度" in clause and (
        has_recent_action(feature_bundle, "下蹲") or has_recent_action(feature_bundle, "伏地")
    ):
        bonus += 0.05
    if "配合探头和走位继续找机会" in clause and sample_type == "Action":
        bonus += 0.10
    if "观察爆炸结果并衔接走位" in clause and sample_type == "Grenade":
        bonus += 0.18
    if "准备衔接枪线压制或继续逼位" in clause and sample_type == "Grenade" and segment_index == 3:
        bonus += 0.12
    if "通过跳跃继续调整身位" in clause and latest_speed >= 80:
        bonus += 0.04
    if "一边救援一边留意周围威胁" in clause and sample_type == "BeingResuce":
        bonus += 0.10
    if "接着推进、防守或者制造新的交战空间" in clause and sample_type == "SkillStart":
        bonus += 0.08
    if "补足资源后准备继续转点或接战" in clause and sample_type == "Looting":
        bonus += 0.10

    return bonus


def should_use_top1_direct(
    query_sample: MatchSample,
    retrieved_labels: list[str],
    retrieved_scores: list[float],
) -> bool:
    if not retrieved_labels or not retrieved_scores:
        return False
    sample_type = query_sample.sample_type
    if sample_type in TOP1_DIRECT_TYPES:
        return True
    threshold = TOP1_DIRECT_THRESHOLDS.get(sample_type, 0.95)
    return float(retrieved_scores[0]) >= threshold


def choose_clause(
    values: list[tuple[str, float]],
    default_clause: str | None,
    sample_type: str,
    segment_index: int,
    feature_bundle: FeatureBundle,
) -> str | None:
    score_map = aggregate_scores(values)
    if default_clause:
        score_map[default_clause] = score_map.get(default_clause, 0.0) + 0.12

    if not score_map:
        return default_clause

    for clause in list(score_map):
        score_map[clause] += clause_context_bonus(clause, sample_type, segment_index, feature_bundle)

    return max(score_map.items(), key=lambda item: item[1])[0]


def render_prediction(
    query_sample: MatchSample,
    feature_bundle: FeatureBundle,
    retrieved_labels: list[str],
    retrieved_scores: list[float],
) -> tuple[str, str]:
    fallback = build_training_label(query_sample, feature_bundle)
    fallback_first, fallback_second, fallback_third = split_label_segments(fallback)

    if not retrieved_labels:
        return fallback, "fallback_template"

    top1_label = retrieved_labels[0]
    top1_first, top1_second, top1_third = split_label_segments(top1_label)

    if should_use_top1_direct(query_sample, retrieved_labels, retrieved_scores):
        return top1_label, "top1_direct"

    weighted_second: list[tuple[str, float]] = []
    weighted_third: list[tuple[str, float]] = []

    for label, score in zip(retrieved_labels, retrieved_scores):
        _, second, third = split_label_segments(label)
        if second:
            weighted_second.append((second, score))
        if third:
            weighted_third.append((third, score))

    first = top1_first or fallback_first or "先做身位和视角调整"
    second = choose_clause(
        weighted_second,
        top1_second or fallback_second,
        query_sample.sample_type,
        2,
        feature_bundle,
    ) or fallback_second or "继续根据信息调整位置"
    third = choose_clause(
        weighted_third,
        top1_third or fallback_third,
        query_sample.sample_type,
        3,
        feature_bundle,
    ) or fallback_third or "最后衔接推进、防守或转移"

    if third == second:
        third = top1_third or fallback_third or "最后衔接推进、防守或转移"
        if third == second:
            third = "最后根据局势继续推进、防守或转移"

    return f"主玩家先{first}，随后{second}，最后{third}。", "hybrid_rerank"
