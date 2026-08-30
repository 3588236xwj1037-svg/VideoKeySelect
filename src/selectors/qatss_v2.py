from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    low = float(values.min())
    high = float(values.max())

    if high - low < 1e-8:
        return np.zeros_like(values)

    return (values - low) / (high - low)


def frame_to_pil(frame) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    array = np.asarray(frame)

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")

    if array.ndim == 3:
        return Image.fromarray(array[..., :3]).convert("RGB")

    raise TypeError(f"Unsupported frame type or shape: {type(frame)}, {array.shape}")


def frame_to_gray(frame) -> np.ndarray:
    image = frame_to_pil(frame)
    image = image.convert("L").resize((32, 32))

    return np.asarray(image, dtype=np.float32) / 255.0


def compute_motion_scores(frames) -> np.ndarray:
    count = len(frames)
    scores = np.zeros(count, dtype=np.float32)

    if count <= 1:
        return scores

    previous = frame_to_gray(frames[0])

    for index in range(1, count):
        current = frame_to_gray(frames[index])
        scores[index] = float(np.mean(np.abs(current - previous)))
        previous = current

    return normalize(scores)


def compute_relevance_scores(
    frames,
    question: str,
    model,
    preprocess,
    tokenizer,
    device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()

    text_tokens = tokenizer([question]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / (
            text_features.norm(dim=-1, keepdim=True) + 1e-12
        )

        all_scores = []

        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]

            images = torch.stack(
                [
                    preprocess(frame_to_pil(frame))
                    for frame in batch_frames
                ]
            ).to(device)

            image_features = model.encode_image(images)
            image_features = image_features / (
                image_features.norm(dim=-1, keepdim=True) + 1e-12
            )

            scores = image_features @ text_features.T
            all_scores.extend(
                scores[:, 0].float().cpu().tolist()
            )

    return normalize(np.asarray(all_scores, dtype=np.float32))


def choose_anchors(
    scores: np.ndarray,
    timestamps,
    count: int,
    min_gap_seconds: float,
) -> list[int]:
    ranking = np.argsort(-scores)
    anchors = []

    for index in ranking:
        index = int(index)

        far_enough = all(
            abs(
                float(timestamps[index])
                - float(timestamps[other])
            ) >= min_gap_seconds
            for other in anchors
        )

        if far_enough:
            anchors.append(index)

        if len(anchors) >= count:
            break

    return anchors


def get_local_window(
    anchor: int,
    total: int,
    window_size: int,
) -> list[int]:
    window_size = min(window_size, total)

    start = anchor - window_size // 2
    start = max(0, min(start, total - window_size))

    return list(range(start, start + window_size))


def qatss_v2_select(
    frames,
    timestamps,
    question: str,
    top_k: int,
    model,
    preprocess,
    tokenizer,
    device,
    local_ratio: float = 0.75,
    window_size: int = 3,
    motion_weight: float = 0.35,
    anchor_gap_seconds: float = 3.0,
    alpha=None,
    min_gap_seconds=None,
):
    total = len(frames)

    if total == 0:
        return [], {}

    top_k = min(int(top_k), total)
    window_size = max(1, min(int(window_size), top_k))

    # ============================================================
    # 阶段一：动作锚点检测
    # 使用问题相关性 + 相邻帧画面变化度计算动作分数。
    # ============================================================
    relevance = compute_relevance_scores(
        frames=frames,
        question=question,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
    )

    motion = compute_motion_scores(frames)

    action_score = (
        (1.0 - motion_weight) * relevance
        + motion_weight * motion
    )

    local_budget = min(
        top_k,
        max(
            window_size,
            int(np.ceil(top_k * local_ratio)),
        ),
    )

    global_budget = top_k - local_budget

    anchor_count = max(
        1,
        int(np.ceil(local_budget / window_size)),
    )

    anchors = choose_anchors(
        scores=action_score,
        timestamps=timestamps,
        count=anchor_count,
        min_gap_seconds=anchor_gap_seconds,
    )

    # ============================================================
    # 阶段二：局部连续窗口保留
    # 围绕动作锚点保留连续帧，避免丢失短时动作过程。
    # ============================================================
    selected_local = []

    for anchor in anchors:
        local_indices = get_local_window(
            anchor=anchor,
            total=total,
            window_size=window_size,
        )

        for index in local_indices:
            if index not in selected_local:
                selected_local.append(index)

            if len(selected_local) >= local_budget:
                break

        if len(selected_local) >= local_budget:
            break

    # 如果局部窗口重叠，按动作分数补齐局部预算。
    for index in np.argsort(-action_score):
        index = int(index)

        if index not in selected_local:
            selected_local.append(index)

        if len(selected_local) >= local_budget:
            break

    selected = list(selected_local)

    # ============================================================
    # 阶段三：全局时间覆盖补充
    # 将剩余帧预算分配到视频的不同时间区域。
    # ============================================================
    selected_global = []

    if global_budget > 0:
        edges = np.linspace(
            0,
            total,
            global_budget + 1,
            dtype=int,
        )

        for bin_index in range(global_budget):
            start = int(edges[bin_index])
            end = int(edges[bin_index + 1])

            if end <= start:
                end = min(total, start + 1)

            candidates = [
                index
                for index in range(start, end)
                if index not in selected
            ]

            if not candidates:
                continue

            best = max(
                candidates,
                key=lambda index: float(relevance[index]),
            )

            selected.append(int(best))
            selected_global.append(int(best))

    # 最终兜底，确保返回 top_k 帧。
    for index in np.argsort(-action_score):
        index = int(index)

        if index not in selected:
            selected.append(index)

        if len(selected) >= top_k:
            break

    selected = sorted(selected[:top_k])

    details = {
        "relevance": relevance.tolist(),
        "novelty": motion.tolist(),
        "motion": motion.tolist(),
        "combined_score": action_score.tolist(),
        "ranking": [
            int(index)
            for index in np.argsort(-action_score)
        ],
        "anchors": [int(index) for index in anchors],
        "selected_local": [
            int(index)
            for index in sorted(selected_local)
        ],
        "selected_global": [
            int(index)
            for index in sorted(selected_global)
        ],
        "local_budget": int(local_budget),
        "global_budget": int(global_budget),
        "window_size": int(window_size),
        "motion_weight": float(motion_weight),
        "anchor_gap_seconds": float(anchor_gap_seconds),
    }

    return selected, details