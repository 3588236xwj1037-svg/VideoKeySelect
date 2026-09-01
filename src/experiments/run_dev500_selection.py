"""Generate a fair dev500 frame-selection comparison for four methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.samplers.candidates import sample_by_fps
from src.samplers.uniform import save_frames
from src.selectors.clip_topk import extract_clip_features, load_clip_model
from src.selectors.qatss import minmax_normalize, temporal_diverse_select
from src.selectors.qatss_v2 import select_indices_from_features


DEFAULT_MANIFEST = Path("data/nextqa/manifests/dev500.jsonl")
DEFAULT_OUTPUT_ROOT = Path("results/dev500_selection_compare")

SAMPLE_FPS = 1.0
TOP_K = 4
RANDOM_SEED = 42
QATSS_V1_ALPHA = 0.75
MIN_GAP_SECONDS = 1.5

METHODS = ("uniform", "clip_topk", "qatss", "qatss_v2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def load_records(manifest_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def uniform_indices(num_frames: int, top_k: int) -> list[int]:
    if num_frames <= 0 or top_k <= 0:
        return []
    return np.linspace(
        0,
        num_frames - 1,
        num=min(top_k, num_frames),
        dtype=int,
    ).tolist()


def qatss_v1_from_features(
    image_features: np.ndarray,
    text_feature: np.ndarray,
    timestamps,
    top_k: int,
    alpha: float,
    min_gap_seconds: float,
) -> tuple[list[int], dict]:
    """Apply the original QATSS v1 scoring using already extracted features."""
    relevance = image_features @ text_feature
    novelty = np.zeros(len(image_features), dtype=np.float32)
    if len(image_features) > 1:
        novelty[1:] = 1.0 - np.sum(
            image_features[1:] * image_features[:-1],
            axis=1,
        )

    normalized_relevance = minmax_normalize(relevance)
    normalized_novelty = minmax_normalize(novelty)
    combined_score = alpha * normalized_relevance + (1.0 - alpha) * normalized_novelty
    selected = temporal_diverse_select(
        scores=combined_score,
        timestamps=timestamps,
        top_k=top_k,
        min_gap_seconds=min_gap_seconds,
    )

    return selected, {
        "relevance": relevance.tolist(),
        "novelty": novelty.tolist(),
        "combined_score": combined_score.tolist(),
        "ranking": np.argsort(-combined_score, kind="stable").tolist(),
        "alpha": float(alpha),
        "min_gap_seconds": float(min_gap_seconds),
    }


def save_method_frames(frames, indices: list[int], output_dir: Path) -> None:
    save_frames([frames[index] for index in indices], output_dir)


def main() -> None:
    args = parse_args()
    records = load_records(args.manifest)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    missing = [
        record["video_path"]
        for record in records
        if not Path(record["video_path"]).exists()
    ]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} video files are missing. First paths:\n{preview}"
        )

    print("Loading CLIP once for all samples...")
    model, preprocess, tokenizer, device = load_clip_model()
    summary = []

    for number, record in enumerate(records, start=1):
        video_id = str(record["video_id"])
        qid = str(record["qid"])
        sample_name = f"{video_id}_{qid}"
        output_dir = args.output_root / sample_name

        print(f"\n[{number}/{len(records)}] {sample_name}")
        print("Question:", record["question"])
        frames, timestamps = sample_by_fps(
            record["video_path"],
            sample_fps=args.sample_fps,
        )
        if not frames:
            raise RuntimeError(f"No candidate frames: {record['video_path']}")

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
        clip_selected = sorted(
            int(index) for index in clip_ranking[: min(args.top_k, len(frames))]
        )

        qatss_v1_selected, qatss_v1_details = qatss_v1_from_features(
            image_features=image_features,
            text_feature=text_feature,
            timestamps=timestamps,
            top_k=args.top_k,
            alpha=QATSS_V1_ALPHA,
            min_gap_seconds=MIN_GAP_SECONDS,
        )
        qatss_v2_selected, qatss_v2_details = select_indices_from_features(
            frame_features=image_features,
            text_feature=text_feature,
            timestamps=timestamps,
            question=record["question"],
            question_type=record.get("type"),
            top_k=args.top_k,
            relevance_weight=0.85,
            min_gap_seconds=MIN_GAP_SECONDS,
            diversity_weight=0.12,
            coverage_weight=0.10,
            local_context_seconds=1.5,
        )

        methods = {
            "uniform": uniform_indices(len(frames), args.top_k),
            "clip_topk": clip_selected,
            "qatss": qatss_v1_selected,
            "qatss_v2": qatss_v2_selected,
        }
        for method, indices in methods.items():
            save_method_frames(frames, indices, output_dir / f"{method}_frames")

        report = {
            **record,
            "candidate_sample_fps": args.sample_fps,
            "candidate_frame_count": len(frames),
            "random_seed": args.seed,
            "top_k": args.top_k,
            "methods": {
                method: {
                    "selected_indices": indices,
                    "selected_timestamps": [timestamps[index] for index in indices],
                }
                for method, indices in methods.items()
            },
            "clip_topk": {
                "relevance_scores": [float(score) for score in clip_scores],
                "ranking": [int(index) for index in clip_ranking],
            },
            "qatss": qatss_v1_details,
            "qatss_v2": qatss_v2_details,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "selection_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            "Times:",
            {
                method: report["methods"][method]["selected_timestamps"]
                for method in METHODS
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

    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(summary)} reports to: {args.output_root}")


if __name__ == "__main__":
    main()
