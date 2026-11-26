r"""
Package the FINAL fusion model (I3D + CLIP) for the cloud service and upload it.

Run from D:\mstct with the (mstct) environment active, after logging in to Hugging Face:
    python -c "from huggingface_hub import login; login()"
    python package_v5.py
Creates D:\mstct\hf_v5\ and uploads it to <your-user>/ms-tct-charades-fusion.
"""
import json
import os
import shutil
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import torch

ROOT = r"D:\mstct"
REPO_ID = "namratha-addagada/ms-tct-charades-fusion"
OLD_REPO = "namratha-addagada/ms-tct-charades"          # for the 157 class names
CKPT = os.path.join(ROOT, "runs", "fused_final", "best_weights.pth")
FEATS = os.path.join(ROOT, "features_fused")
OUT = os.path.join(ROOT, "hf_v5")
WIN, FPS, N_BEST, N_TYPICAL = 256, 3.0, 10, 2

sys.path.insert(0, os.path.join(ROOT, "MS-TCT"))
from MSTCT.MSTCT_Model import MSTCT  # noqa: E402


def average_precision(y, s):
    y = np.asarray(y, bool)
    if not y.any():
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    tp = y[o]
    return float((np.cumsum(tp) / np.arange(1, len(tp) + 1) * tp).sum() / tp.sum())


def main():
    from huggingface_hub import HfApi, hf_hub_download

    ann = json.load(open(os.path.join(ROOT, "MS-TCT", "data", "charades.json")))
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    print(f"Model: epoch {ck['epoch']} | test mAP {ck.get('test_map', 0):.2f} | in_dim {ck['in_feat_dim']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSTCT([256, 384, 576, 864], 3, 8, 8, ck["in_feat_dim"], 512, 157)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()

    def predict(f):
        T = len(f)
        n = -(-T // WIN)
        x = np.zeros((n * WIN, f.shape[1]), np.float32)
        x[:T] = f
        x = torch.from_numpy(x).view(n, WIN, -1).permute(0, 2, 1).to(device)
        with torch.inference_mode():
            logits, _ = model(x)
        return torch.sigmoid(logits).reshape(n * WIN, -1)[:T].float().cpu().numpy()

    # ---- 1. score every test video of demo-friendly length -----------------------
    cands = [v for v, a in ann.items() if a["subset"] == "testing" and a["actions"] and 15 <= a["duration"] <= 40]
    print(f"Scoring {len(cands)} test videos (15-40 s) ...")
    rows = []
    for v in cands:
        s = predict(np.load(os.path.join(FEATS, v + ".npy")).astype(np.float32))
        dur = ann[v]["duration"]
        t = (np.arange(len(s)) + 0.5) * dur / len(s)
        gt = sorted({int(a[0]) for a in ann[v]["actions"]})
        top5 = set(np.argsort(-s.mean(0))[:5].tolist())
        aps = []
        for c in gt:
            lab = np.zeros(len(s), bool)
            for cc, st, en in ann[v]["actions"]:
                if int(cc) == c:
                    lab |= (t >= st) & (t <= en)
            if lab.any() and not lab.all():          # timing only measurable if action is not the whole clip
                aps.append(average_precision(lab, s[:, c]))
        rows.append({"video": v, "correct_top5": len(top5 & set(gt)), "n_gt": len(gt),
                     "timing_ap": float(np.nanmean(aps)) if aps else 0.0, "duration": dur})

    rows.sort(key=lambda r: (r["correct_top5"], r["timing_ap"]), reverse=True)
    best = rows[:N_BEST]
    med = np.median([r["correct_top5"] for r in rows])
    typical = [r for r in rows[N_BEST:] if r["correct_top5"] == med][:N_TYPICAL]
    chosen = best + typical
    print("\nChosen demo videos (best first, last 2 = typical cases):")
    for r in chosen:
        print(f"  {r['video']}  {r['correct_top5']}/5 correct  timing AP {r['timing_ap']:.2f}  {r['duration']:.1f}s")
    hits = np.mean([r["correct_top5"] > 0 for r in rows]) * 100
    print(f"\nAll {len(rows)} scored videos: mean correct in top-5 = {np.mean([r['correct_top5'] for r in rows]):.2f}, "
          f"videos with >=1 correct = {hits:.1f}%")

    # ---- 2. build the package -----------------------------------------------------
    shutil.rmtree(OUT, ignore_errors=True)
    for d in ("samples", "videos", "i3d"):
        os.makedirs(os.path.join(OUT, d))
    samples = {}
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(os.path.join(ROOT, "Charades_v1_480.zip")) as z:
        names = {os.path.basename(n)[:-4]: n for n in z.namelist() if n.lower().endswith(".mp4")}
        for r in chosen:
            v = r["video"]
            np.save(os.path.join(OUT, "samples", v + ".npy"),
                    np.load(os.path.join(FEATS, v + ".npy")).astype(np.float16))
            samples[v] = {"duration": ann[v]["duration"], "actions": ann[v]["actions"]}
            dst = os.path.join(OUT, "videos", v + ".mp4")
            with z.open(names[v]) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
            cap = cv2.VideoCapture(dst)
            cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
            ok, frame = cap.read()
            cap.release()
            if ok:
                h, w = frame.shape[:2]
                cv2.imwrite(os.path.join(OUT, "videos", v + ".jpg"),
                            cv2.resize(frame, (320, round(h * 320 / w))), [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(samples, open(os.path.join(OUT, "samples", "samples.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(OUT, "test_video_scores.json"), "w"))

    slim = {k: v for k, v in ck.items() if k not in ("model_state_dict", "settings")}
    slim["model_state_dict"] = {k: v.half() for k, v in ck["model_state_dict"].items()}
    slim["window"] = WIN
    slim["feature_fps"] = FPS
    torch.save(slim, os.path.join(OUT, "mstct_fusion.pth"))
    shutil.copy(os.path.join(FEATS, "_fusion_stats.npz"), os.path.join(OUT, "fusion_stats.npz"))
    shutil.copy(os.path.join(ROOT, "pytorch-i3d", "models", "rgb_charades.pt"), os.path.join(OUT, "i3d", "rgb_charades.pt"))
    shutil.copy(os.path.join(ROOT, "pytorch-i3d", "pytorch_i3d.py"), os.path.join(OUT, "i3d", "pytorch_i3d.py"))
    shutil.copy(hf_hub_download(OLD_REPO, "charades_classes.json"), os.path.join(OUT, "charades_classes.json"))

    tm, ts = ck.get("test_map", 0), ck.get("sampled_test_map", 0)
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(f"""---
license: mit
tags: [action-detection, video, charades, ms-tct, i3d, clip]
---
# MS-TCT with fused I3D + CLIP features (Charades)

Temporal action detection (157 Charades classes) with the official MS-TCT model (Dai et al., CVPR 2022).

* **Input:** 1536-D per snippet at 3 snippets/s = I3D RGB (Charades-fine-tuned, 1024-D) + CLIP ViT-B/32 (512-D),
  each standardised with `fusion_stats.npz`, I3D resampled onto the CLIP time grid. Window 256 snippets.
* **Training:** 7,027 Charades training videos; epoch chosen on 784 held-out training videos.
* **Official test split (1,814 videos):** per-frame mAP **{tm:.2f}%**, sampled mAP **{ts:.2f}%** (epoch {ck['epoch']}).
* Ablation (same settings, epoch chosen on test as in the official code): CLIP 21.00, I3D 22.72, fusion 24.36.

`i3d/` contains the I3D code and Charades weights from piergiaj/pytorch-i3d (Apache-2.0), used to extract
features from uploaded videos. Original MS-TCT code: https://github.com/dairui01/MS-TCT
""")

    # ---- 3. upload --------------------------------------------------------------
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=OUT, repo_id=REPO_ID, repo_type="model",
                      commit_message=f"Final fusion model (test mAP {tm:.2f}) + best demo videos")
    files = api.list_repo_files(REPO_ID)
    print(f"\nUploaded {len(files)} files -> https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
