import csv
import json
from pathlib import Path


CSV_PATH = Path("external/NExT-QA/dataset/nextqa/val.csv")
OUTPUT_PATH = Path("data/nextqa/manifests/dev10.jsonl")

TYPE_ORDER = ["CH", "CW", "TN", "TC", "DC", "DO", "DL"]


def make_record(row):
    answer_index = int(row["answer"])
    answer_letter = chr(ord("A") + answer_index)

    choices = {
        "A": row["a0"],
        "B": row["a1"],
        "C": row["a2"],
        "D": row["a3"],
        "E": row["a4"],
    }

    video_id = row["video"]

    return {
        "dataset": "NExT-QA",
        "split": "val",
        "video_id": video_id,
        "qid": row["qid"],
        "type": row["type"],
        "video_path": f"data/nextqa/videos/{video_id}.mp4",
        "question": row["question"],
        "choices": choices,
        "correct_answer": answer_letter,
    }


def main():
    with CSV_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    selected = []
    used_videos = set()

    # 先尽量覆盖不同问题类型
    for question_type in TYPE_ORDER:
        for row in rows:
            video_id = row["video"]
            if row["type"] == question_type and video_id not in used_videos:
                selected.append(make_record(row))
                used_videos.add(video_id)
                break

    # 不足 10 条时继续补充不同视频
    for row in rows:
        if len(selected) >= 10:
            break

        video_id = row["video"]
        if video_id not in used_videos:
            selected.append(make_record(row))
            used_videos.add(video_id)

    selected = selected[:10]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in selected:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved: {OUTPUT_PATH}")
    print(f"count: {len(selected)}")

    for record in selected:
        print(
            record["video_id"],
            record["type"],
            record["correct_answer"],
            record["question"],
        )


if __name__ == "__main__":
    main()