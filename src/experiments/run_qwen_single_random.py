import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR = "/root/autodl-tmp/hf-cache"
RESULT_ROOT = Path("results/nextqa_val_4882821564_selection")

QUESTION = (
    "Why did the boy pick up one present from the group of them "
    "and move to the sofa?"
)
CHOICES = {
    "A": "share with the girl",
    "B": "approach lady sitting there",
    "C": "unwrap it",
    "D": "playing with toy train",
    "E": "gesture something",
}
CORRECT_ANSWER = "C"


def get_frame_paths(method):
    frame_dir = RESULT_ROOT / f"{method}_frames"
    paths = sorted(frame_dir.glob("*.jpg"))

    if len(paths) != 4:
        raise RuntimeError(f"{method} needs exactly 4 frames, found {len(paths)}")

    return [str(path.resolve()) for path in paths]


def build_prompt():
    options = "\n".join(f"{letter}. {text}" for letter, text in CHOICES.items())
    return (
        "The four images are video frames in chronological order.\n"
        "Answer the multiple-choice question using only the visual evidence.\n"
        "Output only one letter: A, B, C, D, or E.\n\n"
        f"Question: {QUESTION}\n\n"
        f"{options}\n\n"
        "Answer:"
    )


def parse_answer(text):
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else None


def answer_with_frames(model, processor, frame_paths):
    content = [{"type": "image", "image": path} for path in frame_paths]
    content.append({"type": "text", "text": build_prompt()})

    messages = [{"role": "user", "content": content}]
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

    return raw_answer, parse_answer(raw_answer)


def main():
    print("Loading Qwen2.5-VL-3B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
    )

    with open(RESULT_ROOT / "selection_report.json", encoding="utf-8") as file:
        selection_report = json.load(file)

    report = {
        "model": MODEL_ID,
        "question": QUESTION,
        "choices": CHOICES,
        "correct_answer": CORRECT_ANSWER,
        "methods": {},
    }

    for method in ("uniform", "random", "clip_topk", "qatss"):
        frame_paths = get_frame_paths(method)
        raw_answer, parsed_answer = answer_with_frames(
            model,
            processor,
            frame_paths,
        )

        report["methods"][method] = {
            "frame_paths": frame_paths,
            "selected_timestamps": selection_report[method]["selected_timestamps"],
            "raw_answer": raw_answer,
            "parsed_answer": parsed_answer,
            "is_correct": parsed_answer == CORRECT_ANSWER,
        }

        print(
            f"{method}: raw={raw_answer!r}, "
            f"parsed={parsed_answer!r}, "
            f"correct={parsed_answer == CORRECT_ANSWER}"
        )

    output_path = RESULT_ROOT / "qwen_answers_with_random.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()