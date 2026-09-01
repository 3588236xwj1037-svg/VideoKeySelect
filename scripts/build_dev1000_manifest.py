from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_CSV = Path("external/NExT-QA/dataset/nextqa/val.csv")
DEFAULT_VIDEO_DIR = Path("data/nextqa/videos")
DEFAULT_DEV50 = Path("data/nextqa/manifests/dev50.jsonl")
DEFAULT_DEV500 = Path("data/nextqa/manifests/dev500.jsonl")
DEFAULT_IDS = Path("data/nextqa/manifests/dev1000_video_ids.txt")
DEFAULT_OUTPUT = Path("data/nextqa/manifests/dev1000.jsonl")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return sorted(rows, key=lambda row: (str(row["video"]), str(row["qid"])))


def load_manifest_video_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(str(json.loads(line)["video_id"]))
    return ids


def make_record(row: dict[str, str]) -> dict:
    answer_index = int(row["answer"])
    video_id = str(row["video"])

    return {
        "dataset": "NExT-QA",
        "split": "val",
        "video_id": video_id,
        "qid": str(row["qid"]),
        "type": row["type"],
        "video_path": f"data/nextqa/videos/{video_id}.mp4",
        "question": row["question"],
        "choices": {
            "A": row["a0"],
            "B": row["a1"],
            "C": row["a2"],
            "D": row["a3"],
            "E": row["a4"],
        },
        "correct_answer": chr(ord("A") + answer_index),
    }


def proportional_targets(groups: dict[str, list], count: int) -> dict[str, int]:
    total = sum(len(items) for items in groups.values())

    if count >= total:
        return {
            group: len(items)
            for group, items in groups.items()
        }

    exact = {
        group: len(items) * count / total
        for group, items in groups.items()
    }

    targets = {
        group: int(value)
        for group, value in exact.items()
    }

    remaining = count - sum(targets.values())

    order = sorted(
        groups,
        key=lambda group: (
            exact[group] - targets[group],
            group,
        ),
        reverse=True,
    )

    for group in order[:remaining]:
        targets[group] += 1

    return targets


def prepare_video_ids(args, rows: list[dict[str, str]]) -> None:
    if args.ids.exists() and not args.force:
        raise FileExistsError(
            f"{args.ids} already exists. "
            "Use the existing list, or add --force to intentionally regenerate it."
        )

    dev50_ids = load_manifest_video_ids(args.dev50)
    dev500_ids = load_manifest_video_ids(args.dev500)

    local_ids = {
        path.stem
        for path in args.video_dir.glob("*.mp4")
    }

    excluded_ids = dev50_ids | dev500_ids | local_ids

    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video"])].append(row)

    candidates = [
        video_id
        for video_id in by_video
        if video_id not in excluded_ids
    ]

    candidates.sort(
        key=lambda video_id: (
            -len(by_video[video_id]),
            video_id,
        )
    )

    if len(candidates) < args.video_count:
        raise ValueError(
            f"Only {len(candidates)} new candidate videos are available, "
            f"but {args.video_count} are required."
        )

    selected_ids = candidates[:args.video_count]
    total_questions = sum(len(by_video[video_id]) for video_id in selected_ids)

    if total_questions < args.count:
        raise ValueError(
            f"Selected videos contain only {total_questions} questions, "
            f"but {args.count} are required."
        )

    args.ids.parent.mkdir(parents=True, exist_ok=True)
    args.ids.write_text(
        "".join(f"{video_id}\n" for video_id in selected_ids),
        encoding="utf-8",
    )

    print(f"saved: {args.ids}")
    print(f"new videos: {len(selected_ids)}")
    print(f"available questions in selected videos: {total_questions}")
    print(f"excluded dev50 videos: {len(dev50_ids)}")
    print(f"excluded dev500 videos: {len(dev500_ids)}")
    print(f"excluded existing local videos: {len(local_ids)}")


def build_manifest(args, rows: list[dict[str, str]]) -> None:
    if not args.ids.exists():
        raise FileNotFoundError(
            f"{args.ids} does not exist. Run --prepare first."
        )

    selected_ids = {
        line.strip()
        for line in args.ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    if len(selected_ids) != args.video_count:
        raise ValueError(
            f"Expected {args.video_count} unique video IDs, "
            f"found {len(selected_ids)}."
        )

    dev50_ids = load_manifest_video_ids(args.dev50)
    dev500_ids = load_manifest_video_ids(args.dev500)

    overlap_dev50 = selected_ids & dev50_ids
    overlap_dev500 = selected_ids & dev500_ids

    if overlap_dev50:
        raise ValueError(
            f"dev1000 overlaps dev50 videos: {sorted(overlap_dev50)}"
        )

    if overlap_dev500:
        raise ValueError(
            f"dev1000 overlaps dev500 videos: {sorted(overlap_dev500)}"
        )

    missing = [
        str(args.video_dir / f"{video_id}.mp4")
        for video_id in sorted(selected_ids)
        if not (args.video_dir / f"{video_id}.mp4").is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} video files are missing. First paths:\n"
            + "\n".join(missing[:10])
        )

    available = [
        row
        for row in rows
        if str(row["video"]) in selected_ids
    ]

    if len(available) < args.count:
        raise ValueError(
            f"Only {len(available)} questions are available, "
            f"but {args.count} are required."
        )

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in available:
        groups[row["type"]].append(row)

    targets = proportional_targets(groups, args.count)
    rng = random.Random(args.seed)

    selected = []
    for question_type in sorted(groups):
        selected.extend(
            rng.sample(groups[question_type], targets[question_type])
        )

    selected.sort(
        key=lambda row: (
            str(row["video"]),
            str(row["qid"]),
        )
    )

    records = [make_record(row) for row in selected]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    type_counts = defaultdict(int)
    for record in records:
        type_counts[record["type"]] += 1

    print(f"saved: {args.output}")
    print(f"questions: {len(records)}")
    print(f"videos: {len(selected_ids)}")
    print("types:", dict(sorted(type_counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--dev50", type=Path, default=DEFAULT_DEV50)
    parser.add_argument("--dev500", type=Path, default=DEFAULT_DEV500)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    parser.add_argument("--video-count", type=int, default=100)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)

    args = parser.parse_args()

    if args.prepare == args.build:
        parser.error("Specify exactly one of --prepare or --build.")

    rows = load_rows(args.csv)

    if args.prepare:
        prepare_video_ids(args, rows)
    else:
        build_manifest(args, rows)


if __name__ == "__main__":
    main()