from __future__ import annotations

from .features import FeatureBundle, canonical_action_text
from .parser import MatchSample


PRIMARY_PHRASES = {
    "开火": "开火压制正面的敌人",
    "丢雷": "调整投掷角度并丢出投掷物",
    "救援": "贴近倒地队友并开始救援",
    "搜": "靠近附近物资点进行搜刮",
    "放技能": "释放技能争取视野或战术空间",
    "开镜": "开镜观察敌方动向",
    "动作": "先做身位和视角调整",
}

SECONDARY_ACTION_PHRASES = {
    "开镜观察": "继续保持开镜观察",
    "收镜调整": "短暂收镜重新调整",
    "左探头": "向左探头试探信息",
    "右探头": "向右探头试探信息",
    "回正探头": "回正身位继续观察",
    "下蹲卡位": "下蹲贴掩体稳住枪线",
    "伏地": "主动伏地降低暴露",
    "重新起身": "重新起身寻找角度",
    "跳跃调整": "通过跳跃继续调整身位",
    "换弹": "补一次弹药准备下一轮交火",
    "快速移动": "快速移动寻找更稳的位置",
    "滑铲": "用滑铲拉开身位",
    "缓慢移动": "缓慢移动保持信息控制",
}

FALLBACK_SECONDARY = {
    "开火": "继续维持准星和枪线压力",
    "丢雷": "观察爆炸结果并衔接走位",
    "救援": "一边救援一边留意周围威胁",
    "搜": "快速整理补给并观察周围动静",
    "放技能": "根据技能效果重新调整站位",
    "开镜": "配合探头和走位继续找机会",
    "动作": "继续微调站位争取主动权",
}

FALLBACK_FINAL = {
    "开火": "继续输出火力或者准备下一次射击机会",
    "丢雷": "准备衔接枪线压制或继续逼位",
    "救援": "争取把队友拉起后立刻转移或反打",
    "搜": "补足资源后准备继续转点或接战",
    "放技能": "接着推进、防守或者制造新的交战空间",
    "开镜": "视情况继续压枪、转点或拉开身位",
    "动作": "根据局势继续推进、卡位或转移",
}

PLAN_EMPTY_TOKEN = "<NONE>"

LABEL_SEGMENT_PREFIX = "主玩家先"
LABEL_SEGMENT_MIDDLE = "，随后"
LABEL_SEGMENT_TAIL = "，最后"

CLAUSE_TO_ACTION = {phrase: action for action, phrase in SECONDARY_ACTION_PHRASES.items()}
CLAUSE_TO_ACTION.update(
    {
        "顺势持续输出伤害": "持续输出伤害",
        "同时注意规避来袭火力": "规避来袭火力",
        "利用技能效果继续博弈": "利用技能效果",
        "继续维持准星和枪线压力": "持续输出伤害",
        "根据技能效果重新调整站位": "利用技能效果",
        "观察爆炸结果并衔接走位": "跳跃调整",
        "准备衔接枪线压制或继续逼位": "持续输出伤害",
    }
)


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def extract_future_actions(sample: MatchSample) -> list[str]:
    main_player_id = sample.main_player_id
    actions: list[str] = []

    for event in sample.future_events:
        if event.event_type == "动作" and event.actor_id == main_player_id and len(event.raw_fields) >= 2:
            raw_action = event.raw_fields[1].replace("（决策）", "").strip()
            if raw_action:
                actions.append(canonical_action_text(raw_action))
        elif event.event_type == "玩家造成伤害" and event.actor_id == main_player_id:
            actions.append("持续输出伤害")
        elif event.event_type == "玩家造成伤害" and event.target_id == main_player_id:
            actions.append("规避来袭火力")
        elif event.event_type == "技能生效" and event.actor_id == main_player_id:
            actions.append("利用技能效果")

    return dedupe_keep_order(actions)


def extract_future_action_slots(sample: MatchSample, max_slots: int = 3) -> tuple[str, ...]:
    actions = extract_future_actions(sample)[:max_slots]
    while len(actions) < max_slots:
        actions.append(PLAN_EMPTY_TOKEN)
    return tuple(actions)


def future_action_text(sample: MatchSample, max_slots: int = 3) -> str:
    tokens = [token for token in extract_future_action_slots(sample, max_slots=max_slots) if token != PLAN_EMPTY_TOKEN]
    return " -> ".join(tokens) if tokens else PLAN_EMPTY_TOKEN


def split_label_segments(label: str) -> tuple[str | None, str | None, str | None]:
    if not label.startswith(LABEL_SEGMENT_PREFIX):
        return None, None, None
    if LABEL_SEGMENT_MIDDLE not in label or LABEL_SEGMENT_TAIL not in label:
        return None, None, None

    body = label.removeprefix(LABEL_SEGMENT_PREFIX).rstrip("。")
    first, rest = body.split(LABEL_SEGMENT_MIDDLE, 1)
    second, third = rest.split(LABEL_SEGMENT_TAIL, 1)
    return first.strip(), second.strip(), third.strip()


def infer_future_action_slots_from_label(label: str, max_slots: int = 3) -> tuple[str, ...]:
    _, second, third = split_label_segments(label)
    inferred: list[str] = []
    for clause in (second, third):
        action = CLAUSE_TO_ACTION.get(clause or "")
        if action and action not in inferred:
            inferred.append(action)
    while len(inferred) < max_slots:
        inferred.append(PLAN_EMPTY_TOKEN)
    return tuple(inferred[:max_slots])


def build_training_label(sample: MatchSample, feature_bundle: FeatureBundle | None = None) -> str:
    decision_type = sample.decision_type or "动作"
    primary = PRIMARY_PHRASES.get(decision_type, PRIMARY_PHRASES["动作"])

    future_actions = extract_future_actions(sample)
    secondary_candidates: list[str] = []
    for action in future_actions:
        if action in SECONDARY_ACTION_PHRASES:
            secondary_candidates.append(SECONDARY_ACTION_PHRASES[action])
        elif action == "持续输出伤害":
            secondary_candidates.append("顺势持续输出伤害")
        elif action == "规避来袭火力":
            secondary_candidates.append("同时注意规避来袭火力")
        elif action == "利用技能效果":
            secondary_candidates.append("利用技能效果继续博弈")

    if feature_bundle and feature_bundle.visible_enemy_count >= 2:
        secondary_candidates.append("保持对多名敌人的信息压制")
    if feature_bundle and feature_bundle.teammate_down_count > 0 and decision_type != "救援":
        secondary_candidates.append("兼顾队友状态避免被动减员")

    secondary_candidates = dedupe_keep_order(secondary_candidates)
    secondary = secondary_candidates[0] if secondary_candidates else FALLBACK_SECONDARY.get(decision_type, FALLBACK_SECONDARY["动作"])
    final_clause = secondary_candidates[1] if len(secondary_candidates) > 1 else FALLBACK_FINAL.get(decision_type, FALLBACK_FINAL["动作"])

    return f"主玩家先{primary}，随后{secondary}，最后{final_clause}。"
