import json
from pathlib import Path

import numpy as np

from src.samplers.candidates import sample_by_fps
from src.samplers.random import random_indices
from src.samplers.uniform import save_frames
from src.selectors.clip_topk import extract_clip_features, load_clip_model
from src.selectors.qatss_v2 import select_indices_from_features


MANIFEST_PATH = Path("data/nextqa/manifests/dev50.jsonl")
OUTPUT_ROOT = Path("results/dev50_selection_v2_adaptive")

SAMPLE_FPS = 1.0
TOP_K = 4
RANDOM_SEED = 42
ALPHA = 0.85
MIN_GAP_SECONDS = 1.5
DIVERSITY_WEIGHT = 0.12
COVERAGE_WEIGHT = 0.10
LOCAL_CONTEXT_SECONDS = 1.5


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

        image_features, text_feature = extract_clip_features(
            frames=frames,
            question=record["question"],
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
        )
        clip_scores = image_features @ text_feature
        clip_ranking = np.argsort(-clip_scores, kind="stable").tolist()
        clip_selected = sorted(int(index) for index in clip_ranking[:TOP_K])

        qatss_v2_selected, qatss_v2_details = select_indices_from_features(
            frame_features=image_features,
            text_feature=text_feature,
            timestamps=timestamps,
            question=record["question"],
            question_type=record.get("type"),
            top_k=TOP_K,
            relevance_weight=ALPHA,
            min_gap_seconds=MIN_GAP_SECONDS,
            diversity_weight=DIVERSITY_WEIGHT,
            coverage_weight=COVERAGE_WEIGHT,
            local_context_seconds=LOCAL_CONTEXT_SECONDS,
        )

        methods = {
            "uniform": uniform,
            "random": random,
            "clip_topk": clip_selected,
            "qatss_v2": qatss_v2_selected,
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
            "qatss_v2_config": {
                "relevance_weight": ALPHA,
                "min_gap_seconds": MIN_GAP_SECONDS,
                "diversity_weight": DIVERSITY_WEIGHT,
                "coverage_weight": COVERAGE_WEIGHT,
                "local_context_seconds": LOCAL_CONTEXT_SECONDS,
            },
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
            "qatss_v2": {
                "relevance_scores": [
                    float(score) for score in qatss_v2_details["relevance"]
                ],
                "novelty_scores": [
                    float(score) for score in qatss_v2_details["novelty"]
                ],
                "combined_scores": [
                    float(score)
                    for score in qatss_v2_details["combined_score"]
                ],
                "ranking": qatss_v2_details["ranking"],
                "details": qatss_v2_details,
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