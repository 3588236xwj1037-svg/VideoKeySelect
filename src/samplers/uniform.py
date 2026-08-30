from pathlib import Path

import cv2
import numpy as np


def uniform_sample(video_path: str, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")

    indices = np.linspace(
        0,
        total_frames - 1,
        num=min(num_frames, total_frames),
        dtype=int,
    )

    frames = []
    timestamps = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()

        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(index / fps)

    cap.release()
    return frames, timestamps


def save_frames(frames, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path / f"{i:02d}.jpg"), bgr_frame)