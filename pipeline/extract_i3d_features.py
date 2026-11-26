r"""
Extract I3D features for every Charades video - the feature type used in the MS-TCT paper.

Follows piergiaj/pytorch-i3d (extract_features.py + charades_dataset_full.py):
  * I3D RGB fine-tuned on Charades (models/rgb_charades.pt)
  * frames at 24 fps, centre crop 224, pixels scaled to [-1, 1]
  * one 1024-D feature per 8 frames -> 3 features per second
  * videos longer than 1600 frames processed in overlapping chunks (same as the original)

Run from D:\mstct with the (mstct) environment active (needs D:\mstct\pytorch-i3d):
    python extract_i3d_features.py --limit 20      # quick test
    python extract_i3d_features.py                 # all videos
Safe to stop and restart: finished videos are skipped.
v2: frames are repeated for videos recorded below 24 fps; long-video chunk overlap removed.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile

import cv2
import numpy as np

FPS = 24.0
CROP = 224


def load_frames(path, fps=FPS, crop=CROP):
    """Decode a video at `fps`, centre-crop `crop` px. Returns uint8 [T, crop, crop, 3] (RGB)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    native = cap.get(cv2.CAP_PROP_FPS)
    if not native or native < 1 or native > 240:
        native = 30.0
    frames, idx, next_t = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx / native + 1e-6 >= next_t:
            h, w = frame.shape[:2]
            if min(h, w) < 226:                       # same rule as pytorch-i3d
                s = 226.0 / min(h, w)
                frame = cv2.resize(frame, (round(w * s), round(h * s)))
                h, w = frame.shape[:2]
            y, x = (h - crop) // 2, (w - crop) // 2
            rgb = cv2.cvtColor(frame[y:y + crop, x:x + crop], cv2.COLOR_BGR2RGB)
            # Repeat the frame for every 1/24 s it covers, so videos recorded below 24 fps
            # still give a true 24 fps sequence (as in the official 24 fps Charades frame dump).
            while idx / native + 1e-6 >= next_t:
                frames.append(rgb)
                next_t += 1.0 / fps
        idx += 1
    cap.release()
    if not frames:
        return None
    while len(frames) < 16:                           # I3D needs a minimum clip length
        frames.append(frames[-1])
    return np.stack(frames)


class ZipVideos:
    def __init__(self, zip_path, members):
        self.zip_path, self.members, self.zf = zip_path, members, None
        self.tmp = tempfile.mkdtemp(prefix="charades_i3d_")

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        import torch
        vid, member = self.members[i]
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path)
        tmp_path = os.path.join(self.tmp, f"{vid}_{os.getpid()}.mp4")
        frames = None
        try:
            with self.zf.open(member) as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            frames = load_frames(tmp_path)
        except Exception as e:
            print(f"\n[warn] {vid}: {type(e).__name__}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return vid, (torch.from_numpy(frames) if frames is not None else None)


def collate_one(batch):
    return batch[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=r"D:\mstct\Charades_v1_480.zip")
    ap.add_argument("--out", default=r"D:\mstct\features_i3d")
    ap.add_argument("--i3d_repo", default=r"D:\mstct\pytorch-i3d")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    sys.path.insert(0, args.i3d_repo)
    from pytorch_i3d import InceptionI3d

    os.makedirs(args.out, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        members = sorted((os.path.basename(n)[:-4], n) for n in z.namelist() if n.lower().endswith(".mp4"))
    todo = [(v, m) for v, m in members if not os.path.exists(os.path.join(args.out, f"{v}.npy"))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Videos in zip: {len(members)} | to process now: {len(todo)} | workers: {args.workers}")
    if not todo:
        return

    device = torch.device("cuda")
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(os.path.join(args.i3d_repo, "models", "rgb_charades.pt"), map_location="cpu"))
    i3d.to(device).eval()

    def feats_of(x):            # x: [1, 3, T, 224, 224] float on GPU -> [T', 1024]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            f = i3d.extract_features(x)                 # [1, 1024, T', 1, 1]
        return f.float().squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0).cpu().numpy()

    loader = DataLoader(ZipVideos(args.zip, todo), batch_size=1, num_workers=args.workers,
                        collate_fn=collate_one, prefetch_factor=1 if args.workers else None,
                        persistent_workers=args.workers > 0)
    failed, meta, t0 = [], {}, time.time()
    for vid, frames in tqdm(loader, total=len(todo), unit="video", smoothing=0.05):
        if frames is None:
            failed.append(vid)
            continue
        # uint8 [T,H,W,3] -> float [-1,1] [1,3,T,H,W] on the GPU
        x = frames.to(device, non_blocking=True).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0
        t = x.shape[2]
        if t > 1600:                                    # chunking exactly as in pytorch-i3d
            # same overlapping chunks as pytorch-i3d, but keep each 1600-frame chunk only
            # once (drop the snippets that come from the 48/56-frame context margins)
            parts = []
            for c0 in range(1, t - 56, 1600):
                end = min(t - 1, c0 + 1600 + 56)
                start = max(1, c0 - 48)
                fc = feats_of(x[:, :, start:end])
                lead = (c0 - start) // 8
                parts.append(fc[lead:lead + 1600 // 8])
            f = np.concatenate(parts, axis=0)
        else:
            f = feats_of(x)
        np.save(os.path.join(args.out, f"{vid}.npy"), f.astype(np.float16))
        meta[vid] = {"frames": int(t), "snippets": int(len(f))}
        del x, frames
        torch.cuda.empty_cache()           # release GPU memory so it never spills into shared RAM

    log_path = os.path.join(args.out, "_extraction_log.json")
    old = json.load(open(log_path)) if os.path.exists(log_path) else {"meta": {}, "failed": []}
    old["meta"].update(meta)
    old["failed"] = sorted(set(old["failed"]) - set(meta) | set(failed))
    old["model"] = "pytorch-i3d rgb_charades.pt, Mixed_5c avg-pool, 1024-D, 24 fps input, float16"
    json.dump(old, open(log_path, "w"), indent=1)
    print(f"\nDone: {len(meta)} videos in {(time.time() - t0) / 60:.1f} min | failed: {len(failed)} {failed[:10]}")
    if meta:
        v = next(iter(meta))
        print("Example:", v, meta[v], np.load(os.path.join(args.out, f"{v}.npy")).shape)


if __name__ == "__main__":
    main()
