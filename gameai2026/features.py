from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .parser import Event, MatchSample, maybe_float, normalize_player_id


DEFAULT_NUMERIC_KEYS = [
    "visible_enemy_count",
    "nearest_enemy_distance",
    "nearest_teammate_distance",
    "nearby_loot_count",
    "nearby_high_loot_count",
    "recent_outgoing_hit_count_3s",
    "recent_incoming_hit_count_3s",
    "recent_outgoing_hit_count_5s",
    "recent_incoming_hit_count_5s",
    "recent_action_count_3s",
    "recent_skill_count_5s",
    "recent_teammate_down_count_5s",
    "scope_open_ratio_5s",
    "avg_speed_3s",
    "latest_speed",
]

VISIBLE_RE = re.compile(r"对(\d+)的可见性是（([0-9.]+)%）")


@dataclass(slots=True)
class PlayerSnapshot:
    player_id: str
    team_id: str | None
    x: float
    y: float
    z: float
    speed: float
    scope_state: str
    visibility_blob: str


@dataclass(slots=True)
class FeatureBundle:
    numeric_features: dict[str, float]
    summary_text: str
    latest_scope_state: str
    recent_actions: list[str]
    visible_enemy_count: int
    teammate_down_count: int


def distance3d(a: PlayerSnapshot, b: PlayerSnapshot) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def canonical_action_text(action_text: str) -> str:
    value = action_text.replace("（决策）", "").strip()
    if not value:
        return "未知动作"

    mapping = {
        "开镜": "开镜观察",
        "关镜": "收镜调整",
        "左探头": "左探头",
        "右探头": "右探头",
        "回正探头": "回正探头",
        "蹲": "下蹲卡位",
        "趴下": "伏地",
        "趴": "伏地",
        "站": "重新起身",
        "跳": "跳跃调整",
        "换弹": "换弹",
        "奔跑": "快速移动",
        "滑铲": "滑铲",
        "行走": "缓慢移动",
    }
    return mapping.get(value, value)


def parse_player_snapshot(event: Event, team_map: dict[str, str]) -> PlayerSnapshot | None:
    if event.event_type != "玩家基础信息" or len(event.raw_fields) < 5:
        return None

    player_id = normalize_player_id(event.raw_fields[0])
    x = maybe_float(event.raw_fields[1])
    y = maybe_float(event.raw_fields[2])
    z = maybe_float(event.raw_fields[3])
    speed = maybe_float(event.raw_fields[4])

    if not player_id or x is None or y is None or z is None:
        return None

    scope_state = event.raw_fields[-1] if event.raw_fields and event.raw_fields[-1] in {"开镜", "关镜"} else "未知"
    visibility_blob = ""
    for field in event.raw_fields:
        if "可见性" in field:
            visibility_blob = field
            break

    return PlayerSnapshot(
        player_id=player_id,
        team_id=team_map.get(player_id),
        x=x,
        y=y,
        z=z,
        speed=float(speed or 0.0),
        scope_state=scope_state,
        visibility_blob=visibility_blob,
    )


def parse_loot_point(event: Event) -> tuple[int, float, float, float] | None:
    if event.event_type != "可搜索的散点物资" or len(event.raw_fields) < 4:
        return None
    try:
        quality = int(float(event.raw_fields[0]))
    except ValueError:
        return None
    x = maybe_float(event.raw_fields[1])
    y = maybe_float(event.raw_fields[2])
    z = maybe_float(event.raw_fields[3])
    if x is None or y is None or z is None:
        return None
    return quality, x, y, z


def build_team_map(events: list[Event]) -> dict[str, str]:
    team_map: dict[str, str] = {}
    for event in events:
        if event.event_type != "游戏开始" or len(event.raw_fields) < 2:
            continue
        player_id = normalize_player_id(event.raw_fields[0])
        if player_id:
            team_map[player_id] = event.raw_fields[1]
    return team_map


def bucket_distance(distance: float | None) -> str:
    if distance is None or math.isinf(distance):
        return "none"
    if distance < 10:
        return "0-10"
    if distance < 25:
        return "10-25"
    if distance < 50:
        return "25-50"
    if distance < 100:
        return "50-100"
    return "100+"


def visible_enemy_count(snapshot: PlayerSnapshot | None) -> int:
    if snapshot is None or not snapshot.visibility_blob:
        return 0
    count = 0
    for _, ratio in VISIBLE_RE.findall(snapshot.visibility_blob):
        if float(ratio) > 0.1:
            count += 1
    return count


def build_feature_bundle(sample: MatchSample) -> FeatureBundle:
    team_map = build_team_map(sample.history_events)

    latest_states: dict[str, PlayerSnapshot] = {}
    main_speed_history: list[tuple[float, float]] = []
    scope_window: list[str] = []
    recent_actions: list[str] = []
    recent_skill_names: list[str] = []
    loot_points: list[tuple[int, float, float, float]] = []

    outgoing_hit_count_3s = 0
    incoming_hit_count_3s = 0
    outgoing_hit_count_5s = 0
    incoming_hit_count_5s = 0
    recent_action_count_3s = 0
    recent_skill_count_5s = 0
    teammate_down_count = 0

    main_player_id = sample.main_player_id
    latest_timestamp = max((event.timestamp for event in sample.history_events), default=20.0)
    action_window_start = latest_timestamp - 3.0
    state_window_start = latest_timestamp - 5.0

    for event in sample.history_events:
        snapshot = parse_player_snapshot(event, team_map)
        if snapshot is not None:
            latest_states[snapshot.player_id] = snapshot
            if snapshot.player_id == main_player_id and event.timestamp >= action_window_start:
                main_speed_history.append((event.timestamp, snapshot.speed))
            if snapshot.player_id == main_player_id and event.timestamp >= state_window_start:
                scope_window.append(snapshot.scope_state)
            continue

        loot_point = parse_loot_point(event)
        if loot_point is not None:
            loot_points.append(loot_point)
            continue

        if event.event_type == "动作" and event.actor_id == main_player_id and len(event.raw_fields) >= 2:
            action_text = canonical_action_text(event.raw_fields[1])
            if event.timestamp >= state_window_start:
                recent_actions.append(action_text)
            if event.timestamp >= action_window_start:
                recent_action_count_3s += 1
            continue

        if event.event_type == "技能生效" and event.actor_id == main_player_id:
            skill_name = event.raw_fields[2] if len(event.raw_fields) >= 3 else "技能"
            if event.timestamp >= state_window_start:
                recent_skill_names.append(skill_name)
                recent_skill_count_5s += 1
            continue

        if event.event_type == "玩家造成伤害":
            if event.actor_id == main_player_id and event.timestamp >= state_window_start:
                outgoing_hit_count_5s += 1
            if event.target_id == main_player_id and event.timestamp >= state_window_start:
                incoming_hit_count_5s += 1
            if event.actor_id == main_player_id and event.timestamp >= action_window_start:
                outgoing_hit_count_3s += 1
            if event.target_id == main_player_id and event.timestamp >= action_window_start:
                incoming_hit_count_3s += 1
            continue

        if event.event_type in {"玩家击倒", "玩家死亡"} and event.actor_id:
            victim_team = team_map.get(event.actor_id)
            main_team = team_map.get(main_player_id or "")
            if (
                main_team
                and victim_team == main_team
                and event.actor_id != main_player_id
                and event.timestamp >= state_window_start
            ):
                teammate_down_count += 1

    main_state = latest_states.get(main_player_id or "")
    visible_count = visible_enemy_count(main_state)

    nearest_enemy_distance = math.inf
    nearest_teammate_distance = math.inf
    nearby_loot_count = 0
    nearby_high_loot_count = 0

    if main_state is not None:
        main_team = team_map.get(main_state.player_id)
        for player_id, snapshot in latest_states.items():
            if player_id == main_state.player_id:
                continue
            dist = distance3d(main_state, snapshot)
            if snapshot.team_id == main_team:
                nearest_teammate_distance = min(nearest_teammate_distance, dist)
            else:
                nearest_enemy_distance = min(nearest_enemy_distance, dist)

        for quality, x, y, z in loot_points:
            dist = math.sqrt((main_state.x - x) ** 2 + (main_state.y - y) ** 2 + (main_state.z - z) ** 2)
            if dist <= 25:
                nearby_loot_count += 1
                if quality >= 5:
                    nearby_high_loot_count += 1

    scope_open_ratio = 0.0
    if scope_window:
        scope_open_ratio = sum(1 for state in scope_window if state == "开镜") / len(scope_window)

    avg_speed_3s = 0.0
    if main_speed_history:
        avg_speed_3s = sum(speed for _, speed in main_speed_history) / len(main_speed_history)

    latest_speed = main_state.speed if main_state is not None else 0.0
    latest_scope_state = main_state.scope_state if main_state is not None else "未知"

    while len(recent_actions) > 4:
        recent_actions.pop(0)
    while len(recent_skill_names) > 3:
        recent_skill_names.pop(0)

    numeric_features = {
        "visible_enemy_count": float(visible_count),
        "nearest_enemy_distance": 9999.0 if math.isinf(nearest_enemy_distance) else float(nearest_enemy_distance),
        "nearest_teammate_distance": 9999.0 if math.isinf(nearest_teammate_distance) else float(nearest_teammate_distance),
        "nearby_loot_count": float(nearby_loot_count),
        "nearby_high_loot_count": float(nearby_high_loot_count),
        "recent_outgoing_hit_count_3s": float(outgoing_hit_count_3s),
        "recent_incoming_hit_count_3s": float(incoming_hit_count_3s),
        "recent_outgoing_hit_count_5s": float(outgoing_hit_count_5s),
        "recent_incoming_hit_count_5s": float(incoming_hit_count_5s),
        "recent_action_count_3s": float(recent_action_count_3s),
        "recent_skill_count_5s": float(recent_skill_count_5s),
        "recent_teammate_down_count_5s": float(teammate_down_count),
        "scope_open_ratio_5s": float(scope_open_ratio),
        "avg_speed_3s": float(avg_speed_3s),
        "latest_speed": float(latest_speed),
    }

    tokens = [
        f"样本类型={sample.sample_type}",
        f"决策={sample.decision_type or '未知'}",
        f"可见敌人={visible_count}",
        f"最近敌距={bucket_distance(None if math.isinf(nearest_enemy_distance) else nearest_enemy_distance)}",
        f"最近队友距={bucket_distance(None if math.isinf(nearest_teammate_distance) else nearest_teammate_distance)}",
        f"附近物资={nearby_loot_count}",
        f"高价值物资={nearby_high_loot_count}",
        f"近3秒输出命中={outgoing_hit_count_3s}",
        f"近3秒承伤命中={incoming_hit_count_3s}",
        f"近5秒技能={recent_skill_count_5s}",
        f"近5秒队友倒地={teammate_down_count}",
        f"镜态={latest_scope_state}",
        f"均速桶={int(avg_speed_3s // 25) * 25}",
    ]
    tokens.extend(f"动作={action}" for action in recent_actions)
    tokens.extend(f"技能={skill}" for skill in recent_skill_names)

    return FeatureBundle(
        numeric_features=numeric_features,
        summary_text=" ".join(tokens),
        latest_scope_state=latest_scope_state,
        recent_actions=recent_actions,
        visible_enemy_count=visible_count,
        teammate_down_count=teammate_down_count,
    )
