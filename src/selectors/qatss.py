import numpy as np

from src.selectors.clip_topk import extract_clip_features


def minmax_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    value_range = values.max() - values.min()

    if value_range < 1e-8:
        return np.zeros_like(values)

    return (values - values.min()) / value_range


def temporal_diverse_select(scores, timestamps, top_k, min_gap_seconds):
    """按综合分数贪心选帧，保证选中帧之间至少相隔 min_gap_seconds。"""
    ranked_indices = np.argsort(-scores)
    selected = []

    for index in ranked_indices:
        index = int(index)

        is_far_enough = all(
            abs(timestamps[index] - timestamps[chosen]) >= min_gap_seconds
            for chosen in selected
        )

        if is_far_enough:
            selected.append(index)

        if len(selected) == min(top_k, len(scores)):
            break

    # 视频太短、时间约束过强时，补齐剩余帧。
    if len(selected) < min(top_k, len(scores)):
        for index in ranked_indices:
            index = int(index)
            if index not in selected:
                selected.append(index)

            if len(selected) == min(top_k, len(scores)):
                break

    return sorted(selected)


def qatss_select(
    frames,
    timestamps,
    question: str,
    top_k: int,
    model,
    preprocess,
    tokenizer,
    device: str,
    alpha: float = 0.75,
    min_gap_seconds: float = 1.5,
):
    """
    QATSS = Query-Aware Temporal Semantic Sampling。

    综合分数 = alpha * 问题相关性 + (1 - alpha) * 视觉新颖性。
    """
    if len(frames) != len(timestamps):
        raise ValueError("frames 和 timestamps 的长度必须一致。")

    image_features, text_feature = extract_clip_features(
        frames=frames,
        question=question,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
    )

    relevance = image_features @ text_feature

    novelty = np.zeros(len(frames), dtype=np.float32)
    if len(frames) > 1:
        adjacent_similarity = np.sum(
            image_features[1:] * image_features[:-1],
            axis=1,
        )
        novelty[1:] = 1.0 - adjacent_similarity

    normalized_relevance = minmax_normalize(relevance)
    normalized_novelty = minmax_normalize(novelty)

    combined_score = (
        alpha * normalized_relevance
        + (1.0 - alpha) * normalized_novelty
    )

    selected_indices = temporal_diverse_select(
        scores=combined_score,
        timestamps=timestamps,
        top_k=top_k,
        min_gap_seconds=min_gap_seconds,
    )

    details = {
        "relevance": relevance,
        "novelty": novelty,
        "combined_score": combined_score,
        "ranking": np.argsort(-combined_score).tolist(),
    }

    return selected_indices, details