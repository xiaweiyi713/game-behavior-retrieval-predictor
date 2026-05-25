from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gameai2026.features import canonical_action_text
from gameai2026.label_builder import PLAN_EMPTY_TOKEN, dedupe_keep_order
from gameai2026.parser import (
    SAMPLE_TYPE_TO_DECISION,
    extract_actor_target,
    maybe_float,
    normalize_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment train-records CSV with exact future-action slots extracted from raw files."
    )
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "train_records_full.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "train_records_full_with_future.csv")
    parser.add_argument("--sidecar-output", type=Path, default=ROOT / "outputs" / "future_actions_exact.csv")
    parser.add_argument("--error-log", type=Path, default=ROOT / "outputs" / "future_actions_exact_errors.csv")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--flush-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_slots(actions: list[str], max_slots: int = 3) -> tuple[str, str, str, str]:
    slots = actions[:max_slots]
    while len(slots) < max_slots:
        slots.append(PLAN_EMPTY_TOKEN)
    action_text = " -> ".join(token for token in slots if token != PLAN_EMPTY_TOKEN) or PLAN_EMPTY_TOKEN
    return slots[0], slots[1], slots[2], action_text


def extract_exact_future_actions(sample_path_str: str) -> dict[str, str]:
    sample_path = Path(sample_path_str)
    sample_type = sample_path.parent.name
    file_stem = sample_path.stem
    decision_type = SAMPLE_TYPE_TO_DECISION.get(sample_type, "动作")
    first_actor_id: str | None = None
    main_player_id: str | None = None
    future_events: list[tuple[str, str | None, str | None, str | None]] = []

    def process_stream(lines) -> None:
        nonlocal first_actor_id, main_player_id, decision_type
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 2:
                continue

            timestamp = maybe_float(parts[0])
            if timestamp is None:
                continue

            event_type = normalize_text(parts[1])
            raw_fields = tuple(parts[2:])
            actor_id, target_id = extract_actor_target(event_type, raw_fields)
            if first_actor_id is None and actor_id:
                first_actor_id = actor_id

            raw_action: str | None = None
            if event_type == "动作" and len(raw_fields) >= 2:
                raw_action = normalize_text(raw_fields[1])
                if timestamp >= 20.0 and raw_action.startswith("（决策）") and main_player_id is None and actor_id:
                    main_player_id = actor_id
                    decision_type = raw_action.replace("（决策）", "", 1)
            elif timestamp >= 20.0 and event_type.startswith("（决策）") and event_type != "（决策）" and actor_id:
                if main_player_id is None:
                    main_player_id = actor_id
                    decision_type = event_type.replace("（决策）", "", 1)

            if timestamp > 20.0 or event_type.startswith("（决策）") or (
                event_type == "动作" and raw_action is not None and raw_action.startswith("（决策）")
            ):
                future_events.append((event_type, actor_id, target_id, raw_action))

    try:
        try:
            with sample_path.open("r", encoding="utf-8") as f:
                process_stream(f)
        except (UnicodeDecodeError, MemoryError):
            with sample_path.open("r", encoding="utf-8", errors="ignore") as f:
                process_stream(f)

        if main_player_id is None:
            main_player_id = first_actor_id

        actions: list[str] = []
        for event_type, actor_id, target_id, raw_action in future_events:
            if event_type == "动作" and actor_id == main_player_id and raw_action:
                clean_action = raw_action.replace("（决策）", "").strip()
                if clean_action:
                    actions.append(canonical_action_text(clean_action))
            elif event_type == "玩家造成伤害" and actor_id == main_player_id:
                actions.append("持续输出伤害")
            elif event_type == "玩家造成伤害" and target_id == main_player_id:
                actions.append("规避来袭火力")
            elif event_type == "技能生效" and actor_id == main_player_id:
                actions.append("利用技能效果")

        action_1, action_2, action_3, action_text = make_slots(dedupe_keep_order(actions))
        return {
            "sample_path": sample_path_str,
            "sample_type": sample_type,
            "file_stem": file_stem,
            "decision_type_exact": decision_type,
            "future_action_1": action_1,
            "future_action_2": action_2,
            "future_action_3": action_3,
            "future_action_text": action_text,
            "extract_error": "",
        }
    except Exception as exc:
        return {
            "sample_path": sample_path_str,
            "sample_type": sample_type,
            "file_stem": file_stem,
            "decision_type_exact": decision_type,
            "future_action_1": PLAN_EMPTY_TOKEN,
            "future_action_2": PLAN_EMPTY_TOKEN,
            "future_action_3": PLAN_EMPTY_TOKEN,
            "future_action_text": PLAN_EMPTY_TOKEN,
            "extract_error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    paths = [str(path) for path in df["sample_path"].astype(str).tolist()]
    total = len(paths)

    if not args.resume:
        for path in [args.sidecar_output, args.error_log]:
            if path.exists():
                path.unlink()

    existing_sidecar = pd.DataFrame()
    completed_paths: set[str] = set()
    if args.resume and args.sidecar_output.exists():
        existing_sidecar = pd.read_csv(args.sidecar_output)
        if not existing_sidecar.empty and "sample_path" in existing_sidecar.columns:
            existing_sidecar = existing_sidecar.drop_duplicates(subset=["sample_path"], keep="last")
            completed_paths = set(existing_sidecar["sample_path"].astype(str).tolist())

    pending_paths = [path for path in paths if path not in completed_paths]

    def append_rows(rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        batch_df = pd.DataFrame(rows)
        args.sidecar_output.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.sidecar_output.exists() or args.sidecar_output.stat().st_size == 0
        batch_df.to_csv(
            args.sidecar_output,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig" if write_header else "utf-8",
        )

    batch_rows: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for idx, row in enumerate(
            executor.map(extract_exact_future_actions, pending_paths, chunksize=max(1, args.chunksize)),
            start=1,
        ):
            batch_rows.append(row)
            if args.flush_every > 0 and len(batch_rows) >= args.flush_every:
                append_rows(batch_rows)
                batch_rows = []
            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(f"processed {len(completed_paths) + idx}/{total}")

    append_rows(batch_rows)

    if not args.sidecar_output.exists():
        sidecar_df = existing_sidecar.copy()
    else:
        sidecar_df = pd.read_csv(args.sidecar_output)
    if not sidecar_df.empty and "sample_path" in sidecar_df.columns:
        sidecar_df = sidecar_df.drop_duplicates(subset=["sample_path"], keep="last")
        sidecar_df.to_csv(args.sidecar_output, index=False, encoding="utf-8-sig")

    error_series = sidecar_df["extract_error"].fillna("").astype(str)
    errors_df = sidecar_df[error_series.str.len() > 0].copy()
    args.error_log.parent.mkdir(parents=True, exist_ok=True)
    errors_df.to_csv(args.error_log, index=False, encoding="utf-8-sig")

    merge_cols = [
        "sample_path",
        "future_action_1",
        "future_action_2",
        "future_action_3",
        "future_action_text",
    ]
    base_df = df.drop(
        columns=[col for col in ["future_action_1", "future_action_2", "future_action_3", "future_action_text"] if col in df.columns],
        errors="ignore",
    )
    out_df = base_df.merge(sidecar_df[merge_cols], on="sample_path", how="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"saved {len(out_df)} records to {args.output}")
    print(f"saved sidecar to {args.sidecar_output}")
    print(f"errors: {len(errors_df)}")
    if not out_df.empty:
        print("top future_action_1:")
        print(out_df["future_action_1"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
