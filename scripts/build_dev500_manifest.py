"""Build a video-disjoint dev500 manifest from 50 newly added videos."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATH = Path("external/NExT-QA/dataset/nextqa/val.csv")
DEV50_PATH = Path("data/nextqa/manifests/dev50.jsonl")
VIDEO_DIR = Path("data/nextqa/videos")
VIDEO_IDS_PATH = Path("data/nextqa/manifests/dev500_video_ids.txt")
MANIFEST_PATH = Path("data/nextqa/manifests/dev500.jsonl")


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_dev50_videos() -> set[str]:
    return {
        str(json.loads(line)["video_id"])
        for line in DEV50_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def proportional_targets(groups: dict[str, list], count: int) -> dict[str, int]:
    total = sum(len(items) for items in groups.values())
    exact = {kind: len(items) * count / total for kind, items in groups.items()}
    targets = {kind: int(value) for kind, value in exact.items()}

    remaining = count - sum(targets.values())
    order = sorted(
        groups,
        key=lambda kind: (exact[kind] - targets[kind], kind),
        reverse=True,
    )
    for kind in order[:remaining]:
        targets[kind] += 1

    return targets


def make_record(row: dict[str, str]) -> dict:
    return {
        "dataset": "NExT-QA",
        "split": "val",
        "video_id": str(row["video"]),
        "qid": str(row["qid"]),
        "type": row["type"],
        "video_path": f"data/nextqa/videos/{row['video']}.mp4",
        "question": row["question"],
        "choices": {
            "A": row["a0"],
            "B": row["a1"],
            "C": row["a2"],
            "D": row["a3"],
            "E": row["a4"],
        },
        "correct_answer": chr(ord("A") + int(row["answer"])),
    }


def prepare_video_ids(rows: list[dict[str, str]], args) -> None:
    if VIDEO_IDS_PATH.exists() and not args.force:
        raise FileExistsError(
            f"{VIDEO_IDS_PATH} already exists. "
            "Use --force only when you intentionally want a new list."
        )

    dev50_videos = load_dev50_videos()
    local_videos = {path.stem for path in VIDEO_DIR.glob("*.mp4")}

    candidates = defaultdict(list)
    for row in rows:
        video_id = str(row["video"])
        if video_id not in dev50_videos and video_id not in local_videos:
            candidates[video_id].append(row)

    ranked = sorted(
        candidates,
        key=lambda video_id: (-len(candidates[video_id]), video_id),
    )
    selected = ranked[:args.video_count]
    question_total = sum(len(candidates[video_id]) for video_id in selected)

    if len(selected) != args.video_count:
        raise RuntimeError(f"Only {len(selected)} new videos are available.")
    if question_total < args.question_count:
        raise RuntimeError(
            f"{args.video_count} videos provide only {question_total} questions, "
            f"but {args.question_count} are required."
        )

    VIDEO_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VIDEO_IDS_PATH.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print("saved:", VIDEO_IDS_PATH)
    print("new videos:", len(selected))
    print("available questions:", question_total)
    print("first IDs:", selected[:10])


def build_manifest(rows: list[dict[str, str]], args) -> None:
    if not VIDEO_IDS_PATH.exists():
        raise FileNotFoundError(
            f"{VIDEO_IDS_PATH} does not exist. Run with --prepare first."
        )

    video_ids = {
        line.strip()
        for line in VIDEO_IDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(video_ids) != args.video_count:
        raise RuntimeError(
            f"Expected {args.video_count} video IDs, found {len(video_ids)}."
        )

    overlap = video_ids & load_dev50_videos()
    if overlap:
        raise RuntimeError(f"Overlap with dev50 videos: {sorted(overlap)}")

    missing = [
        video_id
        for video_id in sorted(video_ids)
        if not (VIDEO_DIR / f"{video_id}.mp4").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected videos are missing. First: {missing[:10]}"
        )

    available = [row for row in rows if str(row["video"]) in video_ids]
    if len(available) < args.question_count:
        raise RuntimeError(
            f"Only {len(available)} questions are available, "
            f"need {args.question_count}."
        )

    groups = defaultdict(list)
    for row in available:
        groups[row["type"]].append(row)

    targets = proportional_targets(groups, args.question_count)
    rng = random.Random(args.seed)

    selected = []
    for kind in sorted(groups):
        selected.extend(rng.sample(groups[kind], targets[kind]))

    selected.sort(key=lambda row: (str(row["video"]), str(row["qid"])))
    records = [make_record(row) for row in selected]

    MANIFEST_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    print("saved:", MANIFEST_PATH)
    print("questions:", len(records))
    print("videos:", len(video_ids))
    print("types:", dict(sorted(Counter(row["type"] for row in records).items())))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--video-count", type=int, default=50)
    parser.add_argument("--question-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.video_count <= 0 or args.question_count <= 0:
        raise ValueError("video-count and question-count must be positive.")

    rows = load_rows()
    if args.prepare:
        prepare_video_ids(rows, args)
    else:
        build_manifest(rows, args)


if __name__ == "__main__":
    main()