from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image


def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    cache_dir: str = "/root/autodl-tmp/hf-cache",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.eval()

    return model, preprocess, tokenizer, device


def extract_clip_features(
    frames,
    question: str,
    model,
    preprocess,
    tokenizer,
    device: str,
    batch_size: int = 32,
):
    """返回归一化后的图像特征和问题文本特征。"""
    if not frames:
        raise ValueError("候选帧为空。")

    image_features = []

    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            image_batch = torch.stack(
                [preprocess(Image.fromarray(frame)) for frame in batch_frames]
            ).to(device)

            features = model.encode_image(image_batch)
            image_features.append(F.normalize(features, dim=-1))

        image_features = torch.cat(image_features, dim=0)

        text_tokens = tokenizer([question]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

    return (
        image_features.float().cpu().numpy(),
        text_features.float().cpu().numpy()[0],
    )


def clip_topk_select(
    frames,
    question: str,
    top_k: int,
    model,
    preprocess,
    tokenizer,
    device: str,
    batch_size: int = 32,
):
    image_features, text_feature = extract_clip_features(
        frames=frames,
        question=question,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    scores = image_features @ text_feature
    ranked_indices = np.argsort(-scores)
    selected_indices = sorted(
        int(index) for index in ranked_indices[:min(top_k, len(frames))]
    )

    return selected_indices, scores, ranked_indices.tolist()