"""
Extract our own CLIP features for every Charades video.

  video (Charades_v1_480.zip) -> frames at 3 per second -> CLIP ViT-B/32 -> [T, 512] float16 .npy

Run from D:\\mstct with the (mstct) environment active:
    python extract_clip_features.py

Safe to stop and restart: videos that already have a .npy are skipped.
"""
import argparse
import json
import os
import shutil
import tempfile
import time
import zipfile

import cv2
import numpy as np

FPS = 3.0                      # feature snippets per second (128 snippets ~ 43 s)
SIZE = 224                     # CLIP input resolution
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Video decoding (runs inside DataLoader worker processes)
# --------------------------------------------------------------------------- #
def sample_frames(path, fps=FPS, size=SIZE):
    """Decode a video and keep one frame every 1/fps seconds.
    Returns float32 array [N, 3, size, size], CLIP-normalised, and the duration."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0.0
    native = cap.get(cv2.CAP_PROP_FPS)
    if not native or native < 1 or native > 240:
        native = 30.0
    frames, idx, next_t = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / native
        if t + 1e-6 >= next_t:
            h, w = frame.shape[:2]
            s = size / min(h, w)
            frame = cv2.resize(frame, (max(size, round(w * s)), max(size, round(h * s))),
                               interpolation=cv2.INTER_CUBIC)
            h, w = frame.shape[:2]
            y, x = (h - size) // 2, (w - size) // 2
            rgb = cv2.cvtColor(frame[y:y + size, x:x + size], cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            next_t += 1.0 / fps
        idx += 1
    cap.release()
    if not frames:
        return None, 0.0
    arr = (np.stack(frames).astype(np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(0, 3, 1, 2).copy(), idx / native


class ZipVideos:
    """Torch-style dataset: each item = one video's frames. Each worker opens its own zip handle."""

    def __init__(self, zip_path, members):
        self.zip_path, self.members, self.zf = zip_path, members, None
        self.tmp = tempfile.mkdtemp(prefix="charades_")

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        vid, member = self.members[i]
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path)
        tmp_path = os.path.join(self.tmp, f"{vid}_{os.getpid()}.mp4")
        try:
            with self.zf.open(member) as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            frames, dur = sample_frames(tmp_path)
        except Exception as e:                       # keep going on a bad file
            print(f"\n[warn] {vid}: {type(e).__name__}: {e}")
            frames, dur = None, 0.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return vid, frames, dur


def collate_one(batch):
    return batch[0]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=r"D:\mstct\Charades_v1_480.zip")
    ap.add_argument("--out", default=r"D:\mstct\features_clip")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="only process N videos (for a quick test)")
    args = ap.parse_args()

    import torch
    import open_clip
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    os.makedirs(args.out, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        members = sorted((os.path.basename(n)[:-4], n) for n in z.namelist() if n.lower().endswith(".mp4"))
    todo = [(v, m) for v, m in members if not os.path.exists(os.path.join(args.out, f"{v}.npy"))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Videos in zip: {len(members)} | already done: {len(members) - len(todo) if not args.limit else '-'} "
          f"| to process now: {len(todo)} | workers: {args.workers}")
    if not todo:
        return

    device = "cuda"
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model = model.to(device).eval().half()

    loader = DataLoader(ZipVideos(args.zip, todo), batch_size=1, num_workers=args.workers,
                        collate_fn=collate_one, prefetch_factor=4 if args.workers else None,
                        persistent_workers=args.workers > 0)
    failed, meta, t0 = [], {}, time.time()
    for vid, frames, dur in tqdm(loader, total=len(todo), unit="video", smoothing=0.05):
        if frames is None:
            failed.append(vid)
            continue
        feats = []
        with torch.inference_mode():
            for k in range(0, len(frames), 256):
                x = torch.from_numpy(frames[k:k + 256]).to(device, non_blocking=True).half()
                feats.append(model.encode_image(x).float().cpu())
        f = torch.cat(feats).numpy().astype(np.float16)            # [T, 512]
        np.save(os.path.join(args.out, f"{vid}.npy"), f)
        meta[vid] = {"snippets": int(len(f)), "duration": round(float(dur), 2)}

    # keep a small log of what was produced in this run
    log_path = os.path.join(args.out, "_extraction_log.json")
    old = json.load(open(log_path)) if os.path.exists(log_path) else {"meta": {}, "failed": []}
    old["meta"].update(meta)
    old["failed"] = sorted(set(old["failed"]) - set(meta) | set(failed))
    old["fps"], old["model"] = FPS, "open_clip ViT-B-32 (openai), encode_image, 512-D, float16"
    json.dump(old, open(log_path, "w"), indent=1)

    mins = (time.time() - t0) / 60
    print(f"\nDone: {len(meta)} videos in {mins:.1f} min | failed: {len(failed)} {failed[:10]}")
    if meta:
        v = next(iter(meta))
        print("Example:", v, np.load(os.path.join(args.out, f"{v}.npy")).shape)


if __name__ == "__main__":          # required on Windows for DataLoader workers
    main()
