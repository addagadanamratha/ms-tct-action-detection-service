r"""
Fuse I3D (motion, 1024-D) + CLIP (appearance, 512-D) features -> 1536-D per snippet.

Both streams run at ~3 snippets/second and span the whole video, so the I3D sequence is
linearly resampled onto the CLIP time grid (CLIP length = duration x 3 exactly).

Each stream is standardised per dimension (zero mean, unit variance) using statistics
from TRAINING videos only, so neither stream dominates just because of its scale.

Run from D:\mstct with the (mstct) environment active:
    python fuse_features.py
Then train:
    python train_clip.py --features D:\mstct\features_fused --in_dim 1536 --out D:\mstct\runs\fused
"""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm


def stats(folder, ids, max_videos=1500):
    s, s2, n = None, None, 0
    for v in ids[:max_videos]:
        x = np.load(os.path.join(folder, f"{v}.npy")).astype(np.float64)
        s = x.sum(0) if s is None else s + x.sum(0)
        s2 = (x ** 2).sum(0) if s2 is None else s2 + (x ** 2).sum(0)
        n += len(x)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def resample(x, n):
    """Linearly resample a [T, D] sequence to n steps along time."""
    if len(x) == n:
        return x
    src = np.linspace(0, len(x) - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (src - i0)[:, None].astype(np.float32)
    return x[i0] * (1 - w) + x[i1] * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i3d", default=r"D:\mstct\features_i3d")
    ap.add_argument("--clip", default=r"D:\mstct\features_clip")
    ap.add_argument("--out", default=r"D:\mstct\features_fused")
    ap.add_argument("--split", default=r"D:\mstct\MS-TCT\data\charades.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ann = json.load(open(args.split))
    both = sorted(v for v in ann
                  if os.path.exists(os.path.join(args.i3d, f"{v}.npy"))
                  and os.path.exists(os.path.join(args.clip, f"{v}.npy")))
    train = [v for v in both if ann[v]["subset"] == "training"]
    print(f"Videos with both feature types: {len(both)} (training: {len(train)})")

    print("Computing normalisation statistics from training videos ...")
    mi, si = stats(args.i3d, train)
    mc, sc = stats(args.clip, train)
    np.savez(os.path.join(args.out, "_fusion_stats.npz"), i3d_mean=mi, i3d_std=si, clip_mean=mc, clip_std=sc)

    lens = []
    for v in tqdm(both, unit="video"):
        a = (np.load(os.path.join(args.i3d, f"{v}.npy")).astype(np.float32) - mi) / si
        b = (np.load(os.path.join(args.clip, f"{v}.npy")).astype(np.float32) - mc) / sc
        lens.append(abs(len(a) - len(b)))
        a = resample(a, len(b))
        np.save(os.path.join(args.out, f"{v}.npy"), np.concatenate([a, b], 1).astype(np.float16))

    lens = np.array(lens)
    print(f"Done. Raw length difference I3D vs CLIP (before resampling): mean {lens.mean():.2f}, max {lens.max()}, "
          f"|diff|<=1 for {np.mean(lens <= 1) * 100:.1f}% of videos")
    print("Example:", both[0], np.load(os.path.join(args.out, f"{both[0]}.npy")).shape)


if __name__ == "__main__":
    main()
