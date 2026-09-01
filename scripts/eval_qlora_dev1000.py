import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--selection-root", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--uniform-adapter", required=True)
    p.add_argument("--clip-adapter", required=True)
    p.add_argument("--qatss-adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["uniform", "clip_topk", "qatss"],
        choices=["uniform", "clip_topk", "qatss"],
    )
    p.add_argument("--max-pixels", type=int, default=160000)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def sample_id(row):
    return f"{row['video_id']}_{row['qid']}"


def make_prompt(row):
    choices = {str(k): str(v) for k, v in row["choices"].items()}
    options = "\n".join(f"{x}. {choices[x]}" for x in "ABCDE")
    return (
        "The images are video frames in chronological order.\n"
        "Answer the multiple-choice question using visual evidence.\n"
        "Output only one letter: A, B, C, D, or E.\n\n"
        f"Question: {row['question']}\n\n{options}\n\nAnswer:"
    )


def load_selection(selection_root, row, method):
    report_path = selection_root / sample_id(row) / "selection_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame_dir = report_path.parent / f"{method}_frames"
    frames = sorted(frame_dir.glob("*.jpg"))
    if len(frames) != 4 or not all(p.is_file() for p in frames):
        raise FileNotFoundError(f"{sample_id(row)} {method}: expected 4 frames")
    return [str(p.resolve()) for p in frames], report["methods"][method]["selected_timestamps"]


def get_answer(model, processor, row, frame_paths):
    content = [{"type": "image", "image": path} for path in frame_paths]
    content.append({"type": "text", "text": make_prompt(row)})
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    prompt_length = inputs["input_ids"].shape[1]
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
        )
    raw = processor.batch_decode(
        output_ids[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    match = re.search(r"\b([A-E])\b", raw.upper())
    parsed = match.group(1) if match else None
    return raw, parsed, time.perf_counter() - started


def save_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def main():
    args = parse_args()
    torch.manual_seed(42)

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = Path(args.output)
    selection_root = Path(args.selection_root)

    if args.resume and output.is_file():
        records = json.loads(output.read_text(encoding="utf-8"))
        records_by_id = {record["sample_id"]: record for record in records}
    else:
        records_by_id = {
            sample_id(row): {
                "sample_id": sample_id(row),
                "video_id": str(row["video_id"]),
                "qid": str(row["qid"]),
                "correct_answer": str(row["correct_answer"]).upper(),
                "methods": {},
            }
            for row in rows
        }

    adapters = {
        "uniform": args.uniform_adapter,
        "clip_topk": args.clip_adapter,
        "qatss": args.qatss_adapter,
    }
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["model.visual", "model.visual.merger", "lm_head"],
    )
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    processor.image_processor.max_pixels = args.max_pixels

    for method in args.methods:
        pending = [
            row for row in rows
            if method not in records_by_id[sample_id(row)]["methods"]
        ]
        print(f"\n[{method}] pending: {len(pending)}/{len(rows)}")
        if not pending:
            continue

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map="cuda:0",
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(base, adapters[method], is_trainable=False)
        model.eval()
        model.config.use_cache = True

        for index, row in enumerate(pending, start=1):
            frames, timestamps = load_selection(selection_root, row, method)
            raw, parsed, seconds = get_answer(model, processor, row, frames)
            correct = str(row["correct_answer"]).upper()
            records_by_id[sample_id(row)]["methods"][method] = {
                "adapter_path": str(Path(adapters[method]).resolve()),
                "frame_paths": frames,
                "selected_timestamps": timestamps,
                "raw_answer": raw,
                "parsed_answer": parsed,
                "is_correct": parsed == correct,
                "inference_seconds": round(seconds, 4),
            }
            if index % 10 == 0 or index == len(pending):
                records = [records_by_id[sample_id(row)] for row in rows]
                save_json(output, records)
                print(f"[{method}] {index}/{len(pending)}")

        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    records = [records_by_id[sample_id(row)] for row in rows]
    save_json(output, records)
    summary = {}
    for method in args.methods:
        values = [r["methods"][method]["is_correct"] for r in records]
        summary[method] = {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values),
        }
    summary_path = output.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFinal results:")
    for method, result in summary.items():
        print(f"{method}: {result['correct']}/{result['total']} "
              f"accuracy={result['accuracy']:.4f}")
    print(f"Saved answers: {output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()