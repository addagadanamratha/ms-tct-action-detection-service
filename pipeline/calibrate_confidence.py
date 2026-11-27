r"""
How sure is the model? Find the score above which a top-5 prediction is usually right.

For every held-out training video we take the model's top-5 actions (ranked by the mean score over
the video, exactly as the app does) and record whether each one is in the ground truth.
The "likely" threshold is the lowest score from which predictions are correct at least 50 % of the
time (precision measured in score bands, made monotone). Chosen on held-out videos, reported once
on the official test videos.

Run from D:\mstct:  python calibrate_confidence.py
Writes D:\mstct\hf_v5\confidence_params.json and uploads it to the fusion model repo.
"""
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
WIN, TARGET = 256, 0.5

sys.path.insert(0, os.path.join(ROOT, "MS-TCT"))
from MSTCT.MSTCT_Model import MSTCT  # noqa: E402


def collect(ann, subset, predict):
    scores, correct = [], []
    for v, a in ann.items():
        if a["subset"] != subset or not a["actions"]:
            continue
        s = predict(np.load(os.path.join(FEATS, v + ".npy")).astype(np.float32)).mean(0)
        gt = {int(c) for c, _, _ in a["actions"]}
        for c in np.argsort(-s)[:5]:
            scores.append(float(s[c])); correct.append(int(c) in gt)
    return np.array(scores), np.array(correct)


def pick_threshold(s, y, width=0.04):
    """Lowest t such that every score band [b, b+width) with b >= t has precision >= TARGET."""
    grid = np.arange(0.0, 0.6, 0.01)
    prec = []
    for b in grid:
        m = (s >= b) & (s < b + width)
        prec.append(y[m].mean() if m.sum() >= 30 else np.nan)
    prec = np.array(prec)
    ok = np.where(np.isnan(prec), True, prec >= TARGET)
    for i, b in enumerate(grid):
        if ok[i:].all() and not np.isnan(prec[i]):
            return float(round(b, 2)), grid, prec
    return 0.0, grid, prec


def report(name, s, y, t):
    L = s >= t
    print(f"{name}: {len(s)} top-5 predictions | marked likely {L.mean() * 100:.1f}% "
          f"-> correct {y[L].mean() * 100:.1f}% | uncertain -> correct {y[~L].mean() * 100 if (~L).any() else 0:.1f}%")
    return {"n": int(len(s)), "likely_share": round(float(L.mean()), 4),
            "likely_precision": round(float(y[L].mean()), 4),
            "uncertain_precision": round(float(y[~L].mean()), 4) if (~L).any() else None}


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

    print("Scoring held-out training videos ...")
    sv, yv = collect(ann, "validation", predict)
    print("Scoring official test videos ...")
    st, yt = collect(ann, "testing", predict)

    t, grid, prec = pick_threshold(sv, yv)
    print("\nPrecision of top-5 predictions by score band (held-out):")
    for b, p in zip(grid, prec):
        if not np.isnan(p):
            print(f"  score {b:.2f}-{b + 0.04:.2f}: {p * 100:5.1f}% correct")
    print(f"\n==> 'likely' threshold (chosen on held-out videos): score >= {t:.2f}")
    rv = report("held-out", sv, yv, t)
    rt = report("TEST    ", st, yt, t)

    out = os.path.join(ROOT, "hf_v5", "confidence_params.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"likely_threshold": t, "target_precision": TARGET, "heldout": rv, "test": rt}, open(out, "w"), indent=1)
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=out, path_in_repo="confidence_params.json", repo_id=REPO_ID,
                        repo_type="model", commit_message=f"Confidence threshold {t:.2f} (held-out calibrated)")
    print("Saved + uploaded", out)


if __name__ == "__main__":
    main()
