from __future__ import annotations

import re

import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize without producing NaN for constant inputs."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values

    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-8:
        return np.zeros_like(values)

    return (values - low) / (high - low)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def compute_semantic_transition_scores(
    frame_features: np.ndarray,
) -> np.ndarray:
    """Measure adjacent change in CLIP feature space, not raw pixel space."""
    features = _l2_normalize(frame_features)
    scores = np.zeros(len(features), dtype=np.float32)

    if len(features) > 1:
        similarity = np.sum(features[1:] * features[:-1], axis=1)
        scores[1:] = np.clip(1.0 - similarity, 0.0, 2.0)

    return normalize(scores)


def infer_temporal_question(question: str, question_type: str | None = None) -> bool:
    """Identify questions for which one adjacent context frame is useful."""
    if question_type and question_type.upper().startswith("T"):
        return True

    temporal_terms = (
        "after",
        "before",
        "when",
        "while",
        "then",
        "next",
        "first",
        "last",
        "beginning",
        "middle",
        "end",
        "following",
        "until",
    )
    pattern = r"\b(" + "|".join(temporal_terms) + r")\b"
    return bool(re.search(pattern, question.lower()))


def _time_coverage(
    index: int,
    selected: list[int],
    timestamps: np.ndarray,
) -> float:
    if not selected:
        return 1.0

    duration = float(np.max(timestamps) - np.min(timestamps))
    if duration <= 1e-8:
        return 0.0

    distance = min(abs(float(timestamps[index] - timestamps[other])) for other in selected)
    return float(np.clip(distance / duration, 0.0, 1.0))


def _semantic_diversity(
    index: int,
    selected: list[int],
    features: np.ndarray,
) -> float:
    if not selected:
        return 1.0

    similarity = features[index] @ features[selected].T
    return float(np.clip(1.0 - float(np.max(similarity)), 0.0, 2.0) / 2.0)


def _has_required_gap(
    index: int,
    selected: list[int],
    timestamps: np.ndarray,
    min_gap_seconds: float,
) -> bool:
    return all(
        abs(float(timestamps[index] - timestamps[other])) >= min_gap_seconds
        for other in selected
    )


def _select_temporal_context(
    anchor: int,
    selected: list[int],
    quality: np.ndarray,
    transition: np.ndarray,
    timestamps: np.ndarray,
    context_seconds: float,
) -> int | None:
    candidates = [
        index
        for index in range(len(quality))
        if index not in selected
        and 0.0 < abs(float(timestamps[index] - timestamps[anchor])) <= context_seconds
    ]
    if not candidates:
        return None

    # A context frame must still carry meaningful query evidence. This prevents
    # the local window from being filled with an arbitrary neighboring frame.
    relevance_floor = max(0.35, float(quality[anchor]) - 0.35)
    candidates = [index for index in candidates if quality[index] >= relevance_floor]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda index: (float(quality[index]), float(transition[index]), -index),
    )


def select_indices_from_features(
    frame_features: np.ndarray,
    text_feature: np.ndarray,
    timestamps,
    top_k: int,
    *,
    question: str = "",
    question_type: str | None = None,
    relevance_weight: float = 0.85,
    min_gap_seconds: float = 1.5,
    diversity_weight: float = 0.12,
    coverage_weight: float = 0.10,
    local_context_seconds: float = 1.5,
) -> tuple[list[int], dict]:
    """Select frames with relevance-led, temporally adaptive MMR.

    The previous v2 allocated three of four frames to one local window before
    validating that the window was useful. This version selects one high-query
    anchor, permits at most one adjacent context frame for temporal questions,
    then fills the remaining budget using semantic diversity and time coverage.
    """
    features = np.asarray(frame_features, dtype=np.float32)
    text = np.asarray(text_feature, dtype=np.float32).reshape(-1)
    times = np.asarray(timestamps, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError("frame_features must have shape [num_frames, feature_dim].")
    if len(features) != len(times):
        raise ValueError("frame_features and timestamps must have the same length.")
    if features.shape[1] != len(text):
        raise ValueError("frame_features and text_feature dimensions must match.")
    if len(features) == 0 or top_k <= 0:
        return [], {}

    top_k = min(int(top_k), len(features))
    relevance_weight = float(np.clip(relevance_weight, 0.0, 1.0))
    diversity_weight = float(np.clip(diversity_weight, 0.0, 1.0))
    coverage_weight = float(np.clip(coverage_weight, 0.0, 1.0))
    selection_base_weight = max(0.0, 1.0 - diversity_weight - coverage_weight)
    if selection_base_weight == 0.0:
        raise ValueError("diversity_weight + coverage_weight must be less than 1.")

    normalized_features = _l2_normalize(features)
    normalized_text = _l2_normalize(text[None, :])[0]
    relevance = normalized_features @ normalized_text
    normalized_relevance = normalize(relevance)
    semantic_transition = compute_semantic_transition_scores(normalized_features)

    # Semantic change is only a tie breaker for a relevant frame. In contrast
    # to pixel motion, a camera pan cannot independently become an action cue.
    anchor_score = normalized_relevance * (
        relevance_weight + (1.0 - relevance_weight) * semantic_transition
    )
    ranking = np.argsort(-anchor_score, kind="stable")
    anchor = int(ranking[0])
    selected = [anchor]
    selected_context: list[int] = []
    is_temporal = infer_temporal_question(question, question_type)

    if is_temporal and top_k > 1:
        context = _select_temporal_context(
            anchor=anchor,
            selected=selected,
            quality=anchor_score,
            transition=semantic_transition,
            timestamps=times,
            context_seconds=float(max(0.0, local_context_seconds)),
        )
        if context is not None:
            selected.append(context)
            selected_context.append(context)

    selection_scores: dict[int, float] = {anchor: float(anchor_score[anchor])}
    if selected_context:
        context = selected_context[0]
        selection_scores[context] = float(anchor_score[context])

    while len(selected) < top_k:
        unselected = [index for index in range(len(features)) if index not in selected]
        gap_eligible = [
            index
            for index in unselected
            if _has_required_gap(index, selected, times, min_gap_seconds)
        ]
        candidates = gap_eligible or unselected

        def score(index: int) -> float:
            diversity = _semantic_diversity(index, selected, normalized_features)
            coverage = _time_coverage(index, selected, times)
            return (
                selection_base_weight * float(anchor_score[index])
                + diversity_weight * diversity
                + coverage_weight * coverage
            )

        best = max(candidates, key=lambda index: (score(index), float(anchor_score[index]), -index))
        selected.append(best)
        selection_scores[best] = score(best)

    # Frame indices normally already follow time, but sorting by the supplied
    # timestamps keeps the public result chronological for custom samplers too.
    selected = sorted(selected, key=lambda index: (float(times[index]), index))
    selected_global = [index for index in selected if index != anchor and index not in selected_context]

    details = {
        "relevance": relevance.tolist(),
        "normalized_relevance": normalized_relevance.tolist(),
        "novelty": semantic_transition.tolist(),
        "semantic_transition": semantic_transition.tolist(),
        "combined_score": anchor_score.tolist(),
        "anchor_score": anchor_score.tolist(),
        "ranking": [int(index) for index in ranking],
        "anchors": [anchor],
        "selected_local": sorted(selected_context),
        "selected_global": selected_global,
        "selection_scores": {str(index): score for index, score in selection_scores.items()},
        "is_temporal_question": is_temporal,
        "relevance_weight": relevance_weight,
        "diversity_weight": diversity_weight,
        "coverage_weight": coverage_weight,
        "min_gap_seconds": float(min_gap_seconds),
        "local_context_seconds": float(local_context_seconds),
    }
    return selected, details


def qatss_v2_select(
    frames,
    timestamps,
    question: str,
    top_k: int,
    model,
    preprocess,
    tokenizer,
    device: str,
    alpha: float = 0.85,
    min_gap_seconds: float = 1.5,
    diversity_weight: float = 0.12,
    coverage_weight: float = 0.10,
    local_context_seconds: float = 1.5,
    question_type: str | None = None,
):
    """Run QATSS v2 on video frames using one CLIP feature extraction pass."""
    if len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must have the same length.")
    if not frames or top_k <= 0:
        return [], {}

    from src.selectors.clip_topk import extract_clip_features

    image_features, text_feature = extract_clip_features(
        frames=frames,
        question=question,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
    )
    return select_indices_from_features(
        frame_features=image_features,
        text_feature=text_feature,
        timestamps=timestamps,
        top_k=top_k,
        question=question,
        question_type=question_type,
        relevance_weight=alpha,
        min_gap_seconds=min_gap_seconds,
        diversity_weight=diversity_weight,
        coverage_weight=coverage_weight,
        local_context_seconds=local_context_seconds,
    )