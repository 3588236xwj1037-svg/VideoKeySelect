import json
from pathlib import Path

import numpy as np

from src.samplers.candidates import sample_by_fps
from src.samplers.random import random_indices
from src.samplers.uniform import save_frames
from src.selectors.clip_topk import clip_topk_select, load_clip_model
from src.selectors.qatss import qatss_select


MANIFEST_PATH = Path("data/nextqa/manifests/dev10.jsonl")
OUTPUT_ROOT = Path("results/dev10_selection_seed44")

SAMPLE_FPS = 1.0
TOP_K = 4
RANDOM_SEED = 44
ALPHA = 0.75
MIN_GAP_SECONDS = 1.5


def uniform_indices(num_frames: int, top_k: int) -> list[int]:
    if num_frames <= 0 or top_k <= 0:
        return []

    return np.linspace(
        0,
        num_frames - 1,
        num=min(top_k, num_frames),
        dtype=int,
    ).tolist()


def save_method_frames(frames, indices, output_dir: Path) -> None:
    save_frames(
        [frames[index] for index in indices],
        output_dir,
    )


def load_records():
    return [
        json.loads(line)
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    records = load_records()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP once for all samples...")
    model, preprocess, tokenizer, device = load_clip_model()

    summary = []

    for number, record in enumerate(records, start=1):
        video_id = str(record["video_id"])
        qid = str(record["qid"])
        sample_name = f"{video_id}_{qid}"
        output_dir = OUTPUT_ROOT / sample_name

        print(f"\n[{number}/{len(records)}] {sample_name}")
        print("Question:", record["question"])

        frames, timestamps = sample_by_fps(
            record["video_path"],
            sample_fps=SAMPLE_FPS,
        )

        if not frames:
            raise RuntimeError(f"No candidate frames: {record['video_path']}")

        uniform = uniform_indices(len(frames), TOP_K)
        random = random_indices(
            len(frames),
            TOP_K,
            seed=RANDOM_SEED,
        )

        clip_selected, clip_scores, clip_ranking = clip_topk_select(
            frames=frames,
            question=record["question"],
            top_k=TOP_K,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
        )

        qatss_selected, qatss_details = qatss_select(
            frames=frames,
            timestamps=timestamps,
            question=record["question"],
            top_k=TOP_K,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
            alpha=ALPHA,
            min_gap_seconds=MIN_GAP_SECONDS,
        )

        methods = {
            "uniform": uniform,
            "random": random,
            "clip_topk": clip_selected,
            "qatss": qatss_selected,
        }

        for method, indices in methods.items():
            save_method_frames(
                frames,
                indices,
                output_dir / f"{method}_frames",
            )

        report = {
            **record,
            "candidate_sample_fps": SAMPLE_FPS,
            "candidate_frame_count": len(frames),
            "random_seed": RANDOM_SEED,
            "top_k": TOP_K,
            "qatss_alpha": ALPHA,
            "qatss_min_gap_seconds": MIN_GAP_SECONDS,
            "methods": {
                method: {
                    "selected_indices": indices,
                    "selected_timestamps": [
                        timestamps[index] for index in indices
                    ],
                }
                for method, indices in methods.items()
            },
            "clip_topk": {
                "relevance_scores": [
                    float(score) for score in clip_scores
                ],
                "ranking": clip_ranking,
            },
            "qatss": {
                "relevance_scores": [
                    float(score) for score in qatss_details["relevance"]
                ],
                "novelty_scores": [
                    float(score) for score in qatss_details["novelty"]
                ],
                "combined_scores": [
                    float(score)
                    for score in qatss_details["combined_score"]
                ],
                "ranking": qatss_details["ranking"],
            },
        }

        report_path = output_dir / "selection_report.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            "Times:",
            {
                method: report["methods"][method]["selected_timestamps"]
                for method in methods
            },
        )

        summary.append(
            {
                "video_id": video_id,
                "qid": qid,
                "type": record["type"],
                "candidate_frame_count": len(frames),
                "report_path": str(report_path),
            }
        )

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(summary)} reports to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()