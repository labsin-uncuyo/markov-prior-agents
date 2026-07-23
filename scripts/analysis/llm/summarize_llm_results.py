#!/usr/bin/env python3
"""Summarize LLM/ReAct baseline and Markov-guided experiment artifacts.

By default, this script summarizes the 150-episode LLM experiment reported in
the revised manuscript. Use ``--dataset 30_episodes`` to regenerate summaries
for the earlier exploratory run.

Outputs are written to ``data/processed/llm/<dataset>/``:

- ``action_outcome_summary.csv``: Invalid/Redundant/Effective action counts.
- ``episode_length_summary.csv``: per-episode action counts and end reasons.
- ``agent_summary.csv``: compact per-agent summary.

Score mapping follows the analysis used for the manuscript figures:

- Effective: scores 8 or 10.
- Invalid: score 0.
- Redundant: all other scores, typically 3.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import TextIO

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw" / "llm"
PROCESSED_ROOT = ROOT / "data" / "processed" / "llm"

DATASETS = {
    "150_episodes": {
        "LLM Baseline": {
            "episode_data": RAW_ROOT / "150_episodes" / "baseline" / "episode_data.json.gz",
            "summary": RAW_ROOT / "150_episodes" / "baseline" / "summary.txt",
            "log": RAW_ROOT / "150_episodes" / "baseline" / "llm_react.log.gz",
        },
        "LLM-MC": {
            "episode_data": RAW_ROOT / "150_episodes" / "mc_guided" / "episode_data.json.gz",
            "summary": RAW_ROOT / "150_episodes" / "mc_guided" / "summary.txt",
            "log": RAW_ROOT / "150_episodes" / "mc_guided" / "llm_react.log.gz",
        },
    },
    "30_episodes": {
        "LLM Baseline": {
            "episode_data": RAW_ROOT / "30_episodes" / "baseline" / "episode_data_gemma4b.json",
        },
        "LLM-MC": {
            "episode_data": RAW_ROOT / "30_episodes" / "mc_guided" / "episode_data_MC.json",
        },
    },
}
CATEGORIES = ("Invalid", "Redundant", "Effective")


def category(score: int) -> str:
    if score in (8, 10):
        return "Effective"
    if score == 0:
        return "Invalid"
    return "Redundant"


def pct(count: int | float, total: int | float) -> float:
    return (count / total * 100.0) if total else 0.0


def sd(values: list[float]) -> float:
    """Population standard deviation, matching the LLM run summaries."""
    return pstdev(values) if len(values) > 1 else 0.0


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def is_success(end_reason: str) -> bool:
    return "GoalReached" in end_reason or "Success" in end_reason


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_llm_calls(log_path: Path | None) -> int | None:
    if log_path is None or not log_path.exists():
        return None
    count = 0
    with open_text(log_path) as handle:
        for line in handle:
            if "HTTP Request" in line:
                count += 1
    return count


def parse_wall_clock_seconds(summary_path: Path | None) -> int | None:
    if summary_path is None or not summary_path.exists():
        return None
    text = summary_path.read_text(encoding="utf-8")
    match = re.search(r"Total wall-clock\s*:\s*(\d+)h\s+(\d+)m\s+(\d+)s", text)
    if not match:
        return None
    hours, minutes, seconds = (int(value) for value in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def summarize_dataset(dataset: str) -> None:
    inputs = DATASETS[dataset]
    out_dir = PROCESSED_ROOT / dataset
    action_rows = []
    episode_rows = []
    agent_rows = []
    runtime_rows = []

    for agent, artifacts in inputs.items():
        path = artifacts["episode_data"]
        if not path.exists():
            raise FileNotFoundError(path)
        with open_text(path) as handle:
            episodes = json.load(handle)

        counts = {name: 0 for name in CATEGORIES}
        action_lengths = []
        paper_steps = []
        win_action_lengths = []
        win_paper_steps = []
        success_count = 0

        for idx, episode in enumerate(episodes, start=1):
            evaluation = episode.get("evaluation") or []
            end_reason = episode.get("end_reason", "")
            success = is_success(end_reason)
            action_length = len(evaluation)
            # The action-outcome distribution counts all evaluation entries. For
            # runtime/step summaries in the manuscript, the terminal successful
            # action is excluded, hence one step is subtracted from successful
            # episodes only.
            manuscript_steps = action_length - 1 if success and action_length else action_length
            action_lengths.append(action_length)
            paper_steps.append(manuscript_steps)
            if success:
                success_count += 1
                win_action_lengths.append(action_length)
                win_paper_steps.append(manuscript_steps)

            for score in evaluation:
                counts[category(score)] += 1

            episode_rows.append(
                {
                    "agent": agent,
                    "episode": episode.get("episode", idx),
                    "actions_including_terminal": action_length,
                    "manuscript_steps": manuscript_steps,
                    "end_reason": end_reason,
                    "success": success,
                }
            )

        total_actions = sum(counts.values())
        for name in CATEGORIES:
            action_rows.append(
                {
                    "dataset": dataset,
                    "agent": agent,
                    "category": name,
                    "count": counts[name],
                    "pct": pct(counts[name], total_actions),
                }
            )

        agent_rows.append(
            {
                "dataset": dataset,
                "agent": agent,
                "episodes": len(episodes),
                "success_count": success_count,
                "success_pct": pct(success_count, len(episodes)),
                "actions_including_terminal_mean": mean(action_lengths) if action_lengths else 0.0,
                "actions_including_terminal_sd": sd(action_lengths),
                "manuscript_steps_mean": mean(paper_steps) if paper_steps else 0.0,
                "manuscript_steps_sd": sd(paper_steps),
                "win_actions_including_terminal_mean": mean(win_action_lengths) if win_action_lengths else 0.0,
                "win_actions_including_terminal_sd": sd(win_action_lengths),
                "win_manuscript_steps_mean": mean(win_paper_steps) if win_paper_steps else 0.0,
                "win_manuscript_steps_sd": sd(win_paper_steps),
                "total_actions_including_terminal": total_actions,
                "total_manuscript_steps": sum(paper_steps),
            }
        )

        llm_calls = count_llm_calls(artifacts.get("log"))
        wall_clock_seconds = parse_wall_clock_seconds(artifacts.get("summary"))
        if llm_calls is not None or wall_clock_seconds is not None:
            total_steps = sum(paper_steps)
            runtime_rows.append(
                {
                    "dataset": dataset,
                    "agent": agent,
                    "wall_clock_seconds": wall_clock_seconds,
                    "wall_clock_hours": (wall_clock_seconds / 3600.0) if wall_clock_seconds is not None else None,
                    "llm_calls_from_log": llm_calls,
                    "total_manuscript_steps": total_steps,
                    "seconds_per_episode": (wall_clock_seconds / len(episodes)) if wall_clock_seconds is not None and episodes else None,
                    "seconds_per_step": (wall_clock_seconds / total_steps) if wall_clock_seconds is not None and total_steps else None,
                    "llm_calls_per_step": (llm_calls / total_steps) if llm_calls is not None and total_steps else None,
                }
            )

    write_rows(
        out_dir / "action_outcome_summary.csv",
        ["dataset", "agent", "category", "count", "pct"],
        action_rows,
    )
    write_rows(
        out_dir / "episode_length_summary.csv",
        ["agent", "episode", "actions_including_terminal", "manuscript_steps", "end_reason", "success"],
        episode_rows,
    )
    write_rows(
        out_dir / "agent_summary.csv",
        [
            "dataset",
            "agent",
            "episodes",
            "success_count",
            "success_pct",
            "actions_including_terminal_mean",
            "actions_including_terminal_sd",
            "manuscript_steps_mean",
            "manuscript_steps_sd",
            "win_actions_including_terminal_mean",
            "win_actions_including_terminal_sd",
            "win_manuscript_steps_mean",
            "win_manuscript_steps_sd",
            "total_actions_including_terminal",
            "total_manuscript_steps",
        ],
        agent_rows,
    )
    if runtime_rows:
        write_rows(
            out_dir / "runtime_summary.csv",
            [
                "dataset",
                "agent",
                "wall_clock_seconds",
                "wall_clock_hours",
                "llm_calls_from_log",
                "total_manuscript_steps",
                "seconds_per_episode",
                "seconds_per_step",
                "llm_calls_per_step",
            ],
            runtime_rows,
        )
    print(f"Wrote LLM summaries to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="150_episodes",
        help="LLM dataset to summarize (default: 150_episodes).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Summarize every configured dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = sorted(DATASETS) if args.all else [args.dataset]
    for dataset in datasets:
        summarize_dataset(dataset)


if __name__ == "__main__":
    main()
