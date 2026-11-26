r"""
Diagnose why I3D and CLIP feature lengths differ for some videos.
Run from D:\mstct:   python check_lengths.py
"""
import json
import os
import sys

import numpy as np

ROOT = r"D:\mstct"
ann = json.load(open(os.path.join(ROOT, "MS-TCT", "data", "charades.json")))
lp = os.path.join(ROOT, "features_i3d", "_extraction_log.json")
log = json.load(open(lp))["meta"] if os.path.exists(lp) else {}

rows = []
for v in sorted(ann):
    pi, pc = os.path.join(ROOT, "features_i3d", v + ".npy"), os.path.join(ROOT, "features_clip", v + ".npy")
    if not (os.path.exists(pi) and os.path.exists(pc)):
        continue
    ti = np.load(pi, mmap_mode="r").shape[0]
    tc = np.load(pc, mmap_mode="r").shape[0]
    frames = log.get(v, {}).get("frames", -1)
    rows.append((v, ann[v]["duration"], frames, ti, tc, ti - tc))

d = np.array([r[5] for r in rows])
dur = np.array([r[1] for r in rows])
print(f"videos compared: {len(rows)}")
print(f"I3D - CLIP difference: |d|<=1: {np.mean(np.abs(d) <= 1) * 100:.1f}%  |  "
      f"I3D longer: {np.mean(d > 1) * 100:.1f}%  |  CLIP longer: {np.mean(d < -1) * 100:.1f}%")
long = dur > 67
print(f"videos longer than 67 s (I3D chunking kicks in): {long.sum()}  "
      f"-> mean |diff| {np.abs(d[long]).mean() if long.any() else 0:.1f} vs {np.abs(d[~long]).mean():.2f} for shorter videos")
print("\n10 largest differences:")
print(f"{'video':8} {'duration':>8} {'i3d_frames':>10} {'i3d_snip':>8} {'clip_snip':>9} {'diff':>6} {'expected(dur*3)':>15}")
for r in sorted(rows, key=lambda r: -abs(r[5]))[:10]:
    print(f"{r[0]:8} {r[1]:8.1f} {r[2]:10d} {r[3]:8d} {r[4]:9d} {r[5]:6d} {r[1] * 3:15.0f}")

if "--delete" in sys.argv:
    bad = [r[0] for r in rows if abs(r[5]) > 1]
    for v in bad:
        os.remove(os.path.join(ROOT, "features_i3d", v + ".npy"))
    print(f"\nDeleted {len(bad)} mismatched I3D files -> re-run extract_i3d_features.py to redo only these.")
