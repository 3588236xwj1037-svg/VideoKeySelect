import numpy as np


def random_indices(num_frames: int, top_k: int, seed: int = 42) -> list[int]:
    if num_frames <= 0 or top_k <= 0:
        return []

    count = min(num_frames, top_k)
    rng = np.random.default_rng(seed)
    indices = rng.choice(num_frames, size=count, replace=False)

    return sorted(int(index) for index in indices)