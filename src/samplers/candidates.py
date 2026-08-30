import cv2
import numpy as np


def sample_by_fps(video_path: str, sample_fps: float = 1.0):
    cap = cv2.VideoCapture(video_path)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")

    duration = total_frames / source_fps
    timestamps = np.arange(0, duration, 1.0 / sample_fps)

    frames = []
    valid_timestamps = []

    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            valid_timestamps.append(float(timestamp))

    cap.release()
    return frames, valid_timestamps