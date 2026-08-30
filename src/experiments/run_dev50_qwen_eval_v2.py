import json
import re
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR = "/root/autodl-tmp/hf-cache"
MANIFEST_PATH = Path("data/nextqa/manifests/dev50.jsonl")
SELECTION_ROOT = Path("results/dev50_selection_v2")
OUTPUT_ROOT = Path("results/dev50_qwen_eval_v2")
METHODS = ("qatss_v2",)


def load_records():
    return [
        json.loads(line)
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_prompt(record):
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


def parse_answer(text):
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

    new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
    raw_answer = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return raw_answer, parse_answer(raw_answer), time.perf_counter() - started


def get_frame_paths(sample_dir, method):
    paths = sorted((sample_dir / f"{method}_frames").glob("*.jpg"))
    if len(paths) != 4:
        raise RuntimeError(
            f"{sample_dir.name}/{method} requires 4 frames, found {len(paths)}"
        )
    return [str(path.resolve()) for path in paths]


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_summary(results):
    summary = {}

    for method in METHODS:
        answers = [
            item["methods"][method]
            for item in results
            if "error" not in item["methods"][method]
        ]
        correct = sum(item["is_correct"] for item in answers)

        summary[method] = {
            "evaluated": len(answers),
            "correct": correct,
            "accuracy": correct / len(answers) if answers else None,
            "invalid_output_count": sum(
                item["parsed_answer"] is None for item in answers
            ),
            "mean_inference_seconds": (
                sum(item["inference_seconds"] for item in answers) / len(answers)
                if answers
                else None
            ),
        }

    return summary


def main():
    records = load_records()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    answers_path = OUTPUT_ROOT / "answers.json"
    summary_path = OUTPUT_ROOT / "summary.json"

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
        sample_dir = SELECTION_ROOT / sample_name
        selection = json.loads(
            (sample_dir / "selection_report.json").read_text(encoding="utf-8")
        )

        print(f"\n[{number}/{len(records)}] {sample_name}")
        item = {
            "video_id": record["video_id"],
            "qid": record["qid"],
            "type": record["type"],
            "question": record["question"],
            "correct_answer": record["correct_answer"],
            "methods": {},
        }

        for method in METHODS:
            try:
                content = []

                if method != "text_only":
                    frame_paths = get_frame_paths(sample_dir, method)
                    content.extend(
                        {"type": "image", "image": path}
                        for path in frame_paths
                    )
                    timestamps = selection["methods"][method][
                        "selected_timestamps"
                    ]
                else:
                    frame_paths = []
                    timestamps = []

                content.append({"type": "text", "text": build_prompt(record)})
                messages = [{"role": "user", "content": content}]

                raw, parsed, seconds = generate_answer(
                    model,
                    processor,
                    messages,
                )

                item["methods"][method] = {
                    "frame_paths": frame_paths,
                    "selected_timestamps": timestamps,
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
            f"{method}: "
            f"{stats['correct']}/{stats['evaluated']} "
            f"accuracy={stats['accuracy']}"
        )

    print(f"Saved answers: {answers_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()