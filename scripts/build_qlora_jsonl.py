from __future__ import annotations

import json
from pathlib import Path

TOP_K = 4
METHODS = {
    "uniform": "uniform",
    "clip": "clip_topk",
    "qatss": "qatss",
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_prompt(record: dict) -> str:
    choices = "\n".join(
        f"{letter}. {record['choices'][letter]}" for letter in "ABCDE"
    )
    return (
        "The images are video frames in chronological order.\n"
        "Answer the multiple-choice question using visual evidence.\n"
        "Output only one letter: A, B, C, D, or E.\n\n"
        f"Question: {record['question']}\n\n"
        f"{choices}\n\n"
        "Answer:"
    )


def get_images(record: dict, selection_root: Path, method: str) -> list[str]:
    sample_name = f"{record['video_id']}_{record['qid']}"
    sample_dir = selection_root / sample_name
    report_path = sample_dir / "selection_report.json"

    if not report_path.is_file():
        raise FileNotFoundError(f"缺少选帧报告：{report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        str(report["video_id"]) != str(record["video_id"])
        or str(report["qid"]) != str(record["qid"])
    ):
        raise ValueError(f"报告与 manifest 不匹配：{sample_name}")

    timestamps = report["methods"][method]["selected_timestamps"]
    if len(timestamps) != TOP_K or timestamps != sorted(timestamps):
        raise ValueError(f"{sample_name}/{method} 的时间顺序或帧数异常")

    frame_dir = sample_dir / f"{method}_frames"
    images = sorted(frame_dir.glob("*.jpg"))
    if len(images) != TOP_K:
        raise ValueError(
            f"{sample_name}/{method} 需要 {TOP_K} 张图，实际 {len(images)} 张"
        )
    if not all(image.is_file() for image in images):
        raise FileNotFoundError(f"{sample_name}/{method} 存在缺失帧")

    return [str(image.resolve()) for image in images]


def build_examples(manifest: Path, selection_root: Path, method: str) -> list[dict]:
    examples = []

    for record in load_jsonl(manifest):
        answer = record["correct_answer"]
        if answer not in "ABCDE":
            raise ValueError(f"非法答案标签：{record['video_id']}_{record['qid']}")

        images = get_images(record, selection_root, method)
        examples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>" * TOP_K + "\n" + build_prompt(record),
                    },
                    {"role": "assistant", "content": answer},
                ],
                "images": images,
            }
        )

    return examples


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    splits = {
        "train": (
            Path("data/nextqa/manifests/qlora_train.jsonl"),
            Path("results/qlora_train_selection"),
        ),
        "val": (
            Path("data/nextqa/manifests/qlora_val.jsonl"),
            Path("results/qlora_val_selection"),
        ),
    }

    for split, (manifest, selection_root) in splits.items():
        for short_name, method in METHODS.items():
            examples = build_examples(manifest, selection_root, method)
            output = Path(f"data/processed/qlora_{split}_{short_name}.jsonl")
            write_jsonl(output, examples)
            print(f"{output}: {len(examples)} 条")


if __name__ == "__main__":
    main()