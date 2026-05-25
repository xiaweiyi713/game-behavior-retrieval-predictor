from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


SAMPLE_TYPE_TO_DECISION = {
    "Action": "动作",
    "BeingResuce": "救援",
    "Fire": "开火",
    "Grenade": "丢雷",
    "Looting": "搜",
    "SkillStart": "放技能",
}

DECISION_PREFIXES = ("（决策）", "(决策)")


@dataclass(slots=True)
class Event:
    timestamp: float
    event_type: str
    raw_fields: tuple[str, ...]
    raw_line: str
    actor_id: str | None = None
    target_id: str | None = None

    @property
    def is_explicit_decision(self) -> bool:
        if self.event_type in {"（决策）", "(决策)"}:
            return True
        if any(self.event_type.startswith(prefix) for prefix in DECISION_PREFIXES):
            return True
        if self.event_type == "动作" and len(self.raw_fields) >= 2:
            return self.raw_fields[1].startswith(DECISION_PREFIXES)
        return False


@dataclass(slots=True)
class MatchSample:
    source_path: Path
    sample_type: str
    file_stem: str
    events: list[Event]
    history_events: list[Event]
    future_events: list[Event]
    main_player_id: str | None
    decision_type: str
    target_player_id: str | None


def normalize_text(value: str) -> str:
    return value.strip().replace("(决策)", "（决策）")


def normalize_player_id(token: str | None) -> str | None:
    if token is None:
        return None
    token = token.strip()
    if not token or token == "0":
        return None
    if token.startswith("玩家"):
        token = token[2:]
    return token or None


def maybe_float(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def numeric_suffix(text: str) -> int:
    match = re.search(r"(\d+)(?!.*\d)", text)
    return int(match.group(1)) if match else -1


def iter_dataset_files(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    files = [path for path in root.rglob("*.txt") if path.is_file()]
    return sorted(
        files,
        key=lambda path: (numeric_suffix(path.stem), path.parent.name, path.name),
    )


def extract_player_ids(fields: Iterable[str]) -> list[str]:
    player_ids: list[str] = []
    for field in fields:
        if field.startswith("玩家"):
            player_id = normalize_player_id(field)
            if player_id:
                player_ids.append(player_id)
    return player_ids


def extract_actor_target(event_type: str, raw_fields: tuple[str, ...]) -> tuple[str | None, str | None]:
    player_ids = extract_player_ids(raw_fields)

    if event_type == "游戏开始":
        actor = normalize_player_id(raw_fields[0]) if raw_fields else None
        return actor, None

    if event_type == "动作":
        actor = normalize_player_id(raw_fields[0]) if raw_fields else None
        target = player_ids[1] if len(player_ids) > 1 else None
        return actor, target

    if event_type.startswith("（决策）") or event_type == "（决策）":
        actor = normalize_player_id(raw_fields[0]) if raw_fields else None
        target = player_ids[1] if len(player_ids) > 1 else None
        return actor, target

    actor = player_ids[0] if player_ids else None
    target = player_ids[1] if len(player_ids) > 1 else None
    return actor, target


def parse_event_line(line: str) -> Event:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 2:
        raise ValueError(f"Illegal event line: {line}")

    timestamp = maybe_float(parts[0])
    if timestamp is None:
        raise ValueError(f"Illegal timestamp in line: {line}")

    event_type = normalize_text(parts[1])
    raw_fields = tuple(parts[2:])
    actor_id, target_id = extract_actor_target(event_type, raw_fields)
    return Event(
        timestamp=timestamp,
        event_type=event_type,
        raw_fields=raw_fields,
        raw_line=line,
        actor_id=actor_id,
        target_id=target_id,
    )


def infer_decision_type(events: list[Event], sample_type: str) -> str:
    for event in events:
        if event.event_type.startswith("（决策）") and event.event_type != "（决策）":
            return event.event_type.replace("（决策）", "", 1)
        if event.event_type == "动作" and len(event.raw_fields) >= 2:
            action_text = normalize_text(event.raw_fields[1])
            if action_text.startswith("（决策）"):
                return action_text.replace("（决策）", "", 1)
    return SAMPLE_TYPE_TO_DECISION.get(sample_type, "动作")


def infer_main_player_id(events: list[Event], sample_type: str) -> tuple[str | None, str]:
    decision_type = infer_decision_type(events, sample_type)

    for event in events:
        if event.timestamp < 20.0:
            continue
        if event.event_type.startswith("（决策）") and event.actor_id:
            return event.actor_id, decision_type
        if event.event_type == "动作" and len(event.raw_fields) >= 2:
            action_text = normalize_text(event.raw_fields[1])
            if action_text.startswith("（决策）") and event.actor_id:
                return event.actor_id, decision_type

    for event in events:
        if event.actor_id:
            return event.actor_id, decision_type
    return None, decision_type


def split_history_future(events: list[Event]) -> tuple[list[Event], list[Event]]:
    history_events: list[Event] = []
    future_events: list[Event] = []

    for event in events:
        if event.timestamp > 20.0:
            future_events.append(event)
            continue
        if event.is_explicit_decision:
            future_events.append(event)
            continue
        history_events.append(event)
    return history_events, future_events


def load_sample(path: str | Path) -> MatchSample:
    sample_path = Path(path)
    try:
        with sample_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except (UnicodeDecodeError, MemoryError):
        with sample_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
    events = [parse_event_line(line) for line in lines]
    history_events, future_events = split_history_future(events)
    main_player_id, decision_type = infer_main_player_id(events, sample_path.parent.name)

    target_player_id = None
    for event in future_events:
        if event.actor_id == main_player_id and event.target_id:
            target_player_id = event.target_id
            break
        if event.event_type.startswith("（决策）") and event.target_id:
            target_player_id = event.target_id
            break

    return MatchSample(
        source_path=sample_path,
        sample_type=sample_path.parent.name,
        file_stem=sample_path.stem,
        events=events,
        history_events=history_events,
        future_events=future_events,
        main_player_id=main_player_id,
        decision_type=decision_type,
        target_player_id=target_player_id,
    )
