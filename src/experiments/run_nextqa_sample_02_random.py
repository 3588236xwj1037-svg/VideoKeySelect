import json
from pathlib import Path

import numpy as np
from src.samplers.random import random_indices
from src.samplers.candidates import sample_by_fps
from src.samplers.uniform import save_frames
from src.selectors.clip_topk import clip_topk_select, load_clip_model
from src.selectors.qatss import qatss_select


def uniform_indices(num_frames: int, top_k: int):
    return np.linspace(
        0,
        num_frames - 1,
        num=min(top_k, num_frames),
        dtype=int,
    ).tolist()


def save_method_frames(frames, indices, output_dir):
    save_frames(
        [frames[index] for index in indices],
        output_dir,
    )


def main():
    video_path = "data/nextqa/videos/4882821564.mp4"
    question = (
        "why did the boy pick up one present from the group of them "
        "and move to the sofa"
    )
    sample_fps = 1.0
    top_k = 4
    random_seed = 42
    alpha = 0.75
    min_gap_seconds = 1.5

    output_root = Path("results/nextqa_val_4882821564_selection")
    output_root.mkdir(parents=True, exist_ok=True)

    frames, timestamps = sample_by_fps(
        video_path,
        sample_fps=sample_fps,
    )

    print("加载 CLIP...")
    model, preprocess, tokenizer, device = load_clip_model()

    uniform = uniform_indices(len(frames), top_k)
    random = random_indices(len(frames), top_k, seed=random_seed)

    clip_selected, clip_scores, clip_ranking = clip_topk_select(
        frames=frames,
        question=question,
        top_k=top_k,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
    )

    qatss_selected, qatss_details = qatss_select(
        frames=frames,
        timestamps=timestamps,
        question=question,
        top_k=top_k,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        alpha=alpha,
        min_gap_seconds=min_gap_seconds,
    )

    save_method_frames(frames, uniform, output_root / "uniform_frames")
    save_method_frames(frames, clip_selected, output_root / "clip_topk_frames")
    save_method_frames(frames, qatss_selected, output_root / "qatss_frames")
    save_method_frames(frames, random, output_root / "random_frames")

    report = {
        "video_path": video_path,
        "question": question,
        "device": device,
        "candidate_sample_fps": sample_fps,
        "candidate_timestamps": timestamps,
        "top_k": top_k,
        "random_seed": random_seed,
        "qatss_alpha": alpha,
        "qatss_min_gap_seconds": min_gap_seconds,
        "uniform": {
            "selected_indices": uniform,
            "selected_timestamps": [timestamps[index] for index in uniform],
        },
        "random": {
            "selected_indices": random,
            "selected_timestamps": [timestamps[index] for index in random],
        },
        "clip_topk": {
            "selected_indices": clip_selected,
            "selected_timestamps": [
                timestamps[index] for index in clip_selected
            ],
            "relevance_scores": [float(score) for score in clip_scores],
            "ranking": clip_ranking,
        },
        "qatss": {
            "selected_indices": qatss_selected,
            "selected_timestamps": [
                timestamps[index] for index in qatss_selected
            ],
            "relevance_scores": [
                float(score) for score in qatss_details["relevance"]
            ],
            "novelty_scores": [
                float(score) for score in qatss_details["novelty"]
            ],
            "combined_scores": [
                float(score) for score in qatss_details["combined_score"]
            ],
            "ranking": qatss_details["ranking"],
        },
    }

    report_path = output_root / "selection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n实验记录已保存:", report_path)
    print("Uniform:", report["uniform"]["selected_timestamps"])
    print("Random:", report["random"]["selected_timestamps"])
    print("CLIP Top-K:", report["clip_topk"]["selected_timestamps"])
    print("QATSS:", report["qatss"]["selected_timestamps"])


if __name__ == "__main__":
    main()