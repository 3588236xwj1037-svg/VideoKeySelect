"""Evaluate four frame-selection methods on a fixed dev500 manifest."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR = "/root/autodl-tmp/hf-cache"
DEFAULT_MANIFEST = Path("data/nextqa/manifests/dev500.jsonl")
DEFAULT_SELECTION_ROOT = Path("results/dev500_selection_compare")
DEFAULT_OUTPUT_ROOT = Path("results/dev500_qwen_eval_compare")
METHODS = ("uniform", "clip_topk", "qatss", "qatss_v2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def load_records(manifest_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_prompt(record: dict) -> str:
    options = "\n".join(
        f"{letter}. {record['choices'][letter]}"
        for letter in "ABCDE"
    )
    return (
        "The images are video frames in chronological order.\n"
        "Answer the multiple-choice question using visual evidence.\n"
        "Output only one letter: A, B, C, D, or E.\n\n"
        f"Question: {record['question']}\n\n"
        f"{options}\n\n"
        "Answer:"
    )


def parse_answer(text: str) -> str | None:
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else None


def generate_answer(model, processor, messages):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=8,
        )

    new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
    raw_answer = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return raw_answer, parse_answer(raw_answer), time.perf_counter() - started


def get_frame_paths(sample_dir: Path, method: str, top_k: int) -> list[str]:
    paths = sorted((sample_dir / f"{method}_frames").glob("*.jpg"))
    if len(paths) != top_k:
        raise RuntimeError(
            f"{sample_dir.name}/{method} requires {top_k} frames, found {len(paths)}"
        )
    return [str(path.resolve()) for path in paths]


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_summary(results: list[dict]) -> dict:
    summary = {}
    for method in METHODS:
        answers = [
            item["methods"][method]
            for item in results
            if "error" not in item["methods"].get(method, {})
        ]
        by_type = defaultdict(list)
        for item in results:
            method_result = item["methods"].get(method, {})
            if "error" not in method_result:
                by_type[item["type"]].append(method_result)

        summary[method] = {
            "evaluated": len(answers),
            "correct": sum(item["is_correct"] for item in answers),
            "accuracy": (
                sum(item["is_correct"] for item in answers) / len(answers)
                if answers
                else None
            ),
            "invalid_output_count": sum(
                item["parsed_answer"] is None for item in answers
            ),
            "mean_inference_seconds": (
                sum(item["inference_seconds"] for item in answers) / len(answers)
                if answers
                else None
            ),
            "by_type": {
                item_type: {
                    "evaluated": len(type_answers),
                    "correct": sum(item["is_correct"] for item in type_answers),
                    "accuracy": (
                        sum(item["is_correct"] for item in type_answers)
                        / len(type_answers)
                        if type_answers
                        else None
                    ),
                }
                for item_type, type_answers in sorted(by_type.items())
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    records = load_records(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    answers_path = args.output_root / "answers.json"
    summary_path = args.output_root / "summary.json"

    print("Loading Qwen2.5-VL-3B once...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    results = []
    for number, record in enumerate(records, start=1):
        sample_name = f"{record['video_id']}_{record['qid']}"
        sample_dir = args.selection_root / sample_name
        selection_path = sample_dir / "selection_report.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        item = {
            "video_id": record["video_id"],
            "qid": record["qid"],
            "type": record["type"],
            "question": record["question"],
            "correct_answer": record["correct_answer"],
            "methods": {},
        }
        print(f"\n[{number}/{len(records)}] {sample_name}")

        for method in METHODS:
            try:
                frame_paths = get_frame_paths(sample_dir, method, args.top_k)
                content = [
                    {"type": "image", "image": path}
                    for path in frame_paths
                ]
                content.append({"type": "text", "text": build_prompt(record)})
                raw, parsed, seconds = generate_answer(
                    model,
                    processor,
                    [{"role": "user", "content": content}],
                )
                item["methods"][method] = {
                    "frame_paths": frame_paths,
                    "selected_timestamps": selection["methods"][method][
                        "selected_timestamps"
                    ],
                    "raw_answer": raw,
                    "parsed_answer": parsed,
                    "is_correct": parsed == record["correct_answer"],
                    "inference_seconds": round(seconds, 3),
                }
                print(
                    f"{method}: {parsed!r}, "
                    f"correct={parsed == record['correct_answer']}"
                )
            except Exception as error:
                item["methods"][method] = {"error": repr(error)}
                print(f"{method}: ERROR {error!r}")

        results.append(item)
        write_json(answers_path, results)

    summary = build_summary(results)
    write_json(summary_path, summary)
    print("\nFinal results:")
    for method, stats in summary.items():
        print(
            f"{method}: {stats['correct']}/{stats['evaluated']} "
            f"accuracy={stats['accuracy']}"
        )
    print(f"Saved answers: {answers_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
