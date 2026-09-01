from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path("/root/autodl-tmp/video-keyframe-vqa")
VIDEO_DIR = ROOT / "data/nextqa/videos"
MANIFEST_DIR = ROOT / "data/nextqa/manifests"

ID_FILES = [
    MANIFEST_DIR / "qlora_train_video_ids.txt",
    MANIFEST_DIR / "qlora_val_video_ids.txt",
]

def load_ids():
    ids = set()
    for path in ID_FILES:
        ids.update(
            x.strip()
            for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        )
    return sorted(ids)

def download(video_id):
    target = VIDEO_DIR / f"{video_id}.mp4"

    if target.is_file() and target.stat().st_size > 0:
        return video_id, True, "exists"

    try:
        hf_hub_download(
            repo_id="VLM2Vec/nextqa-rawvideo",
            repo_type="dataset",
            filename=f"{video_id}.mp4",
            local_dir=VIDEO_DIR,
        )

        if target.is_file() and target.stat().st_size > 0:
            return video_id, True, "downloaded"

        return video_id, False, "file missing after download"

    except Exception as error:
        return video_id, False, repr(error)

def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ids = load_ids()
    failed = []

    print("待处理视频:", len(ids))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, video_id) for video_id in ids]

        for index, future in enumerate(as_completed(futures), 1):
            video_id, ok, message = future.result()

            if ok:
                print(f"[{index}/{len(ids)}] {message}: {video_id}")
            else:
                failed.append(video_id)
                print(f"[{index}/{len(ids)}] failed: {video_id}")
                print(message[:300])

    failed_path = MANIFEST_DIR / "qlora_missing_video_ids.txt"
    failed_path.write_text(
        "\n".join(sorted(failed)) + ("\n" if failed else ""),
        encoding="utf-8",
    )

    print("下载失败:", len(failed))
    print("失败清单:", failed_path)

if __name__ == "__main__":
    main()