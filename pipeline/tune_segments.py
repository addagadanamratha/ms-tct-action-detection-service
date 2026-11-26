r"""
Tune how per-snippet scores are turned into [start, end] action segments.

The rule (same as the server): smooth the class score curve with a window of `k` snippets,
mark snippets with score >= max(abs_thr, rel_thr * peak), merge gaps <= `gap` seconds,
drop pieces shorter than `min_len` seconds.

Parameters are chosen on the 784 HELD-OUT training videos only, then measured once on the
official test videos. Metric: frame-level F1 between predicted and ground-truth intervals,
averaged over every (video, class) where the class is in the model's top 5 AND in the
ground truth (exactly what the app draws).

Run from D:\mstct:  python tune_segments.py
Writes D:\mstct\hf_v5\segment_params.json and uploads it to the fusion model repo.
"""
import itertools
import json
import os
import sys

import numpy as np
import torch

ROOT = r"D:\mstct"
REPO_ID = "namratha-addagada/ms-tct-charades-fusion"
CKPT = os.path.join(ROOT, "runs", "fused_final", "best_weights.pth")
SPLIT = os.path.join(ROOT, "runs", "fused_final", "_holdout_split.json")
FEATS = os.path.join(ROOT, "features_fused")
WIN = 256
OLD = dict(k=3, rel=0.5, abs=0.05, gap=1.0, min_len=0.6)        # what the server uses now

sys.path.insert(0, os.path.join(ROOT, "MS-TCT"))
from MSTCT.MSTCT_Model import MSTCT  # noqa: E402


def segments(curve, step, k, rel, abs_, gap, min_len):
    if k > 1:
        c = np.convolve(np.pad(curve, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, "valid")
    else:
        c = curve
    on = c >= max(abs_, rel * c.max())
    segs, start = [], None
    for i, a in enumerate(list(on) + [False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            s, e = start * step, i * step
            if segs and s - segs[-1][1] <= gap:
                segs[-1][1] = e
            else:
                segs.append([s, e])
            start = None
    return [(s, e) for s, e in segs if e - s >= min_len]


def frame_f1(pred, gt, dur, res=0.1):
    t = np.arange(0, dur, res)
    p = np.zeros(len(t), bool); g = np.zeros(len(t), bool)
    for s, e in pred:
        p |= (t >= s) & (t < e)
    for s, e in gt:
        g |= (t >= s) & (t < e)
    tp = (p & g).sum()
    if tp == 0:
        return 0.0
    prec, rec = tp / p.sum(), tp / g.sum()
    return 2 * prec * rec / (prec + rec)


def main():
    ann = json.load(open(SPLIT))
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSTCT([256, 384, 576, 864], 3, 8, 8, ck["in_feat_dim"], 512, 157)
    model.load_state_dict(ck["model_state_dict"]); model.to(device).eval()

    def predict(f):
        T = len(f); n = -(-T // WIN)
        x = np.zeros((n * WIN, f.shape[1]), np.float32); x[:T] = f
        x = torch.from_numpy(x).view(n, WIN, -1).permute(0, 2, 1).to(device)
        with torch.inference_mode():
            lg, _ = model(x)
        return torch.sigmoid(lg).reshape(n * WIN, -1)[:T].float().cpu().numpy()

    def cases(subset):
        """(curve, step, duration, gt_intervals) for top-5 classes that are in the ground truth."""
        out = []
        for v, a in ann.items():
            if a["subset"] != subset or not a["actions"]:
                continue
            s = predict(np.load(os.path.join(FEATS, v + ".npy")).astype(np.float32))
            dur = a["duration"]; step = dur / len(s)
            gt = {}
            for c, st, en in a["actions"]:
                gt.setdefault(int(c), []).append((st, en))
            for c in np.argsort(-s.mean(0))[:5]:
                if int(c) in gt:
                    out.append((s[:, c], step, dur, gt[int(c)]))
        return out

    print("Predicting held-out training videos (selection) ...")
    val = cases("validation")
    print("Predicting official test videos (reported once) ...")
    test = cases("testing")
    print(f"(video, class) pairs: held-out {len(val)} | test {len(test)}")

    def S(data, p):
        q = dict(k=p["k"], rel=p["rel"], abs_=p["abs"], gap=p["gap"], min_len=p["min_len"])
        return float(np.mean([frame_f1(segments(c, st, **q), g, d) for c, st, d, g in data]))

    grid = [dict(k=k, rel=r, abs=a, gap=g, min_len=m) for k, r, a, g, m in itertools.product(
        [1, 3, 5, 9], [0.15, 0.25, 0.35, 0.45, 0.55], [0.02, 0.05, 0.1], [1.0, 2.0, 4.0], [0.3, 1.0])]
    sub = val[::3]                                   # coarse search on a third of the held-out pairs
    print(f"Trying {len(grid)} settings ...")
    top10 = sorted(grid, key=lambda p: S(sub, p), reverse=True)[:10]
    best = max(top10, key=lambda p: S(val, p))       # confirm on all held-out pairs

    old_v, new_v = S(val, OLD), S(val, best)
    old_t, new_t = S(test, OLD), S(test, best)
    print("\nHand-picked rule :", OLD)
    print("Tuned rule       :", best)
    print(f"Segment F1, held-out videos : {old_v:.3f} -> {new_v:.3f}")
    print(f"Segment F1, TEST videos     : {old_t:.3f} -> {new_t:.3f}   (reported once)")

    os.makedirs(os.path.join(ROOT, "hf_v5"), exist_ok=True)
    out = os.path.join(ROOT, "hf_v5", "segment_params.json")
    json.dump({**best, "heldout_f1": round(new_v, 4), "test_f1": round(new_t, 4),
               "old_rule": OLD, "old_test_f1": round(old_t, 4)}, open(out, "w"), indent=1)
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=out, path_in_repo="segment_params.json", repo_id=REPO_ID,
                        repo_type="model", commit_message=f"Tuned segment rule (test F1 {new_t:.3f})")
    print("Saved + uploaded", out)


if __name__ == "__main__":
    main()
