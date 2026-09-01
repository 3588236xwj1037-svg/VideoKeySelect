import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = "ABCDE"


def get_value(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def clean_video_id(value):
    value = str(value).strip().split("/")[-1]

    if value.lower().endswith(".mp4"):
        value = value[:-4]

    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]

    return value


def normalize_answer(value):
    value = str(value).strip().upper()

    if value in LETTERS:
        return value

    if value.isdigit() and 0 <= int(value) < len(LETTERS):
        return LETTERS[int(value)]

    raise ValueError(f"无法解析答案: {value}")


def load_excluded_video_ids(paths):
    excluded = set()

    for path in paths:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"找不到排除清单: {path}")

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                excluded.add(str(row["video_id"]))

    return excluded


def build_record(row, index):
    video_id = clean_video_id(
        get_value(row, "video", "video_id", "vid")
    )

    question = get_value(row, "question")
    answer = get_value(row, "answer", "correct_answer", "label")
    qid = get_value(row, "qid", "question_id") or str(index)
    sample_type = get_value(row, "type")

    if not video_id or not question or not answer:
        return None

    choices = {}

    for index, letter in enumerate(LETTERS):
        choice = get_value(
            row,
            f"a{index}",
            f"A{index}",
            f"choice_{index}",
        )

        if not choice:
            return None

        choices[letter] = choice

    try:
        correct_answer = normalize_answer(answer)
    except ValueError:
        return None

    return {
        "dataset": "NExT-QA",
        "split": "train",
        "video_id": video_id,
        "qid": qid,
        "type": sample_type,
        "video_path": f"data/nextqa/videos/{video_id}.mp4",
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
    }


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        ) + "\n",
        encoding="utf-8",
    )


def write_ids(path, ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(sorted(ids)) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="external/NExT-QA/dataset/nextqa/train.csv",
    )
    parser.add_argument(
        "--train-videos",
        type=int,
        default=180,
    )
    parser.add_argument(
        "--val-videos",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--train-output",
        default="data/nextqa/manifests/qlora_train.jsonl",
    )
    parser.add_argument(
        "--val-output",
        default="data/nextqa/manifests/qlora_val.jsonl",
    )
    parser.add_argument(
        "--train-id-output",
        default="data/nextqa/manifests/qlora_train_video_ids.txt",
    )
    parser.add_argument(
        "--val-id-output",
        default="data/nextqa/manifests/qlora_val_video_ids.txt",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[
            "data/nextqa/manifests/dev50.jsonl",
            "data/nextqa/manifests/dev500.jsonl",
            "data/nextqa/manifests/dev1000.jsonl",
        ],
    )

    args = parser.parse_args()

    if args.train_videos <= 0 or args.val_videos <= 0:
        raise ValueError("train-videos 和 val-videos 必须大于 0")

    excluded = load_excluded_video_ids(args.exclude)

    groups = defaultdict(list)

    with Path(args.csv).open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = csv.DictReader(file)

        for index, row in enumerate(rows):
            record = build_record(row, index)

            if record is None:
                continue

            video_id = record["video_id"]

            if video_id in excluded:
                continue

            groups[video_id].append(record)

    candidate_ids = list(groups.keys())
    random.Random(args.seed).shuffle(candidate_ids)

    required = args.train_videos + args.val_videos

    if len(candidate_ids) < required:
        raise RuntimeError(
            f"可用训练视频只有 {len(candidate_ids)} 个，"
            f"但需要 {required} 个"
        )

    selected_ids = candidate_ids[:required]

    val_ids = set(selected_ids[:args.val_videos])
    train_ids = set(selected_ids[args.val_videos:])

    if train_ids & val_ids:
        raise AssertionError("训练集和验证集存在视频重合")

    if train_ids & excluded:
        raise AssertionError("训练集与 dev 集存在视频重合")

    if val_ids & excluded:
        raise AssertionError("验证集与 dev 集存在视频重合")

    train_rows = []
    val_rows = []

    for video_id in sorted(train_ids):
        for row in groups[video_id]:
            row = dict(row)
            row["split"] = "train"
            train_rows.append(row)

    for video_id in sorted(val_ids):
        for row in groups[video_id]:
            row = dict(row)
            row["split"] = "val"
            val_rows.append(row)

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.val_output, val_rows)
    write_ids(args.train_id_output, train_ids)
    write_ids(args.val_id_output, val_ids)

    print("训练视频数:", len(train_ids))
    print("验证视频数:", len(val_ids))
    print("训练问题数:", len(train_rows))
    print("验证问题数:", len(val_rows))
    print("训练题型:", dict(Counter(row["type"] for row in train_rows)))
    print("验证题型:", dict(Counter(row["type"] for row in val_rows)))
    print("dev 视频排除数:", len(excluded))
    print("训练集:", args.train_output)
    print("验证集:", args.val_output)


if __name__ == "__main__":
    main()