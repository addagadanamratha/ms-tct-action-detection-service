r"""
Train MS-TCT on our own CLIP features (full Charades training set).

Uses the OFFICIAL MS-TCT model, data loader and loss (from D:\\mstct\\MS-TCT),
with the settings from the authors' run_MSTCT_Charades.sh:
    batch 32, num_clips 256, lr 1e-4, Adam, 50 epochs, alpha_l 1, beta_l 0.05,
    ReduceLROnPlateau(factor 0.5, patience 8)
Only the input size changes: 512 (CLIP ViT-B/32) instead of 1024 (I3D).

Run from D:\\mstct with the (mstct) environment active:
    python train_clip.py
Quick smoke test (2 epochs, 300 videos):
    python train_clip.py --epochs 2 --limit 300
Regularised run (less overfitting):
    python train_clip.py --out D:\mstct\runs\clip_reg --epochs 30 --opt adamw --wd 0.05 ^
        --sched cosine --snip_drop 0.1 --feat_noise 0.1 --ema 0.999
"""
import math
import argparse
import csv
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MS-TCT")
sys.path.insert(0, REPO)

from apmeter import APMeter                                   # noqa: E402
from charades_dataloader import Charades, collate_fn_unisize  # noqa: E402
from utils import focal_loss, sampled_25, mask_probs          # noqa: E402
from MSTCT.MSTCT_Model import MSTCT                           # noqa: E402

NUM_CLASSES = 157


def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def subset_json(src, n, out):
    """Small json with the first n training + n//4 testing videos (for smoke tests)."""
    data = json.load(open(src))
    tr = [v for v in data if data[v]["subset"] == "training"][:n]
    te = [v for v in data if data[v]["subset"] == "testing"][: max(20, n // 4)]
    json.dump({v: data[v] for v in tr + te}, open(out, "w"))
    return out


def augment(inputs, snip_drop=0.0, feat_noise=0.0):
    """Training-only regularisation on the feature sequence [B, D, T]."""
    if snip_drop > 0:      # hide random snippets -> model must use temporal context
        keep = (torch.rand(inputs.size(0), 1, inputs.size(2), device=inputs.device) >= snip_drop).float()
        inputs = inputs * keep
    if feat_noise > 0:     # small Gaussian noise relative to the feature scale
        inputs = inputs + torch.randn_like(inputs) * feat_noise * inputs.detach().std()
    return inputs


def holdout_json(src, frac, seed, out):
    """Move a fixed random `frac` of TRAINING videos into a 'validation' subset.
    The official test split is untouched and is only reported, never used for selection."""
    data = json.load(open(src))
    train = sorted(v for v in data if data[v]["subset"] == "training")
    rng = random.Random(seed)
    val = set(rng.sample(train, int(round(frac * len(train)))))
    for v in val:
        data[v] = dict(data[v], subset="validation")
    json.dump(data, open(out, "w"))
    return out, len(val)


def run_network(model, batch, device, alpha, beta, aug=None):
    """Same computation as run_network() in the official train.py (+ optional augmentation)."""
    inputs, mask, labels, other, hm = batch
    inputs = inputs.to(device, non_blocking=True).squeeze(3).squeeze(3)   # [B, D, T]
    if aug:
        inputs = augment(inputs, **aug)
    mask = mask.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    hm = hm.to(device, non_blocking=True)

    outputs_final, out_hm = model(inputs)
    probs_f = torch.sigmoid(outputs_final) * mask.unsqueeze(2)
    loss_h = focal_loss(out_hm, hm)
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, reduction="sum") / torch.sum(mask)
    return probs_f, alpha * loss_f + beta * loss_h


def evaluate(model, loader, device, alpha, beta):
    """Official metrics: Full-val-map (all frames) and sampled-val-map (25 points/video)."""
    model.eval()
    apm, sampled = APMeter(), APMeter()
    tot_loss, n, full_probs = 0.0, 0, {}
    with torch.inference_mode():
        for batch in loader:
            probs, loss = run_network(model, batch, device, alpha, beta)
            p = probs.cpu().numpy()[0]; l = batch[2].numpy()[0]; m = batch[1].numpy()[0]
            if m.sum() > 25:
                p1, l1 = sampled_25(p, l, m)
                sampled.add(p1, l1)
            apm.add(p, l)
            full_probs[batch[3][0][0]] = mask_probs(p, m).squeeze().T
            tot_loss += float(loss); n += 1
    ap, sap = 100 * apm.value(), 100 * sampled.value()
    full_map = float(ap.sum() / max(int(torch.nonzero(ap).size(0)), 1))
    samp_map = float(sap.sum() / max(int(torch.nonzero(sap).size(0)), 1))
    return full_map, samp_map, tot_loss / max(n, 1), full_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=r"D:\mstct\features_clip")
    ap.add_argument("--split", default=os.path.join(REPO, "data", "charades.json"))
    ap.add_argument("--out", default=r"D:\mstct\runs\clip_full")
    ap.add_argument("--in_dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_clips", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha_l", type=float, default=1.0)
    ap.add_argument("--beta_l", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only N training videos (smoke test)")
    ap.add_argument("--opt", choices=["adam", "adamw"], default="adam")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay (with --opt adamw)")
    ap.add_argument("--sched", choices=["plateau", "cosine"], default="plateau")
    ap.add_argument("--warmup", type=int, default=2, help="warm-up epochs for cosine schedule")
    ap.add_argument("--snip_drop", type=float, default=0.0, help="prob. of hiding a snippet in training")
    ap.add_argument("--feat_noise", type=float, default=0.0, help="Gaussian noise (x feature std) in training")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights, e.g. 0.999 (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="fraction of training videos held out for model selection (e.g. 0.1)")
    ap.add_argument("--patience", type=int, default=0, help="stop after N epochs without improvement (0 = off)")
    args = ap.parse_args()

    set_seed(args.seed)
    print('Settings:', vars(args))
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    split = subset_json(args.split, args.limit, os.path.join(args.out, "_subset.json")) if args.limit else args.split
    if args.holdout > 0:
        split, n_hold = holdout_json(split, args.holdout, 1234, os.path.join(args.out, "_holdout_split.json"))
        print(f"Held out {n_hold} training videos for model selection (seed 1234); test split only reported.")

    collate = collate_fn_unisize(args.num_clips).charades_collate_fn_unisize
    train_ds = Charades(split, "training", args.features, args.batch_size, NUM_CLASSES, args.num_clips, 0)
    test_ds = Charades(split, "testing", args.features, 1, NUM_CLASSES, args.num_clips, 0)
    sel_ds = Charades(split, "validation", args.features, 1, NUM_CLASSES, args.num_clips, 0) \
        if args.holdout > 0 else test_ds
    if args.holdout > 0:
        sel_ds.split = "testing"   # official loader only crops 'training'/'testing'; use the deterministic test crop
    print(f"Training videos: {len(train_ds)} | selection set: {len(sel_ds)} "
          f"({'held-out training videos' if args.holdout > 0 else 'official test split'}) | test: {len(test_ds)}")

    kw = dict(num_workers=args.workers, pin_memory=True, collate_fn=collate,
              persistent_workers=args.workers > 0)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kw)
    val_dl = torch.utils.data.DataLoader(sel_ds, batch_size=1, shuffle=False, **kw)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, **kw) if args.holdout > 0 else None

    model = MSTCT([256, 384, 576, 864], 3, 8, 8, args.in_dim, 512, NUM_CLASSES).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    if args.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.sched == "cosine":
        def lr_factor(e):    # e = epochs completed
            if e < args.warmup:
                return (e + 1) / (args.warmup + 1)
            prog = (e - args.warmup) / max(1, args.epochs - args.warmup)
            return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    ema = None
    if args.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    aug = {"snip_drop": args.snip_drop, "feat_noise": args.feat_noise} \
        if (args.snip_drop > 0 or args.feat_noise > 0) else None

    log_path = os.path.join(args.out, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_map", "sampled_val_map",
                                "test_map", "sampled_test_map", "lr", "epoch_s"])

    best, best_epoch, bad, best_test, t_start = -1.0, 0, 0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            _, loss = run_network(model, batch, device, args.alpha_l, args.beta_l, aug)
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += float(loss.detach()); n += 1
        eval_model = ema.module if ema is not None else model
        val_map, samp_map, val_loss, probs = evaluate(eval_model, val_dl, device, args.alpha_l, args.beta_l)
        if test_dl is not None:
            test_map, test_samp, _, test_probs = evaluate(eval_model, test_dl, device, args.alpha_l, args.beta_l)
        else:
            test_map, test_samp, test_probs = val_map, samp_map, probs
        if args.sched == "cosine":
            sched.step()
        else:
            sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(tot / max(n, 1), 4), round(val_loss, 4),
                                    round(val_map, 3), round(samp_map, 3), round(test_map, 3), round(test_samp, 3),
                                    lr, round(dt, 1)])

        msg = (f"epoch {epoch:2d}/{args.epochs} | train loss {tot / max(n, 1):.4f} | val loss {val_loss:.4f} "
               f"| val mAP {val_map:6.2f} | sampled mAP {samp_map:6.2f}"
               + (f" | TEST mAP {test_map:6.2f} / {test_samp:6.2f}" if test_dl is not None else "")
               + f" | lr {lr:.1e} | {dt:5.1f}s")
        state = {"model_state_dict": eval_model.state_dict(), "epoch": epoch, "settings": vars(args), "validation_map": val_map,
                 "sampled_validation_map": samp_map, "in_feat_dim": args.in_dim, "num_classes": NUM_CLASSES,
                 "window": args.num_clips, "feature_fps": 3.0,
                 "features": "open_clip ViT-B-32-quickgelu (openai), 512-D",
                 "trained_on": f"Charades full train split ({len(train_ds)} videos), official test split eval"}
        torch.save(state, os.path.join(args.out, "last_weights.pth"))
        state["test_map"], state["sampled_test_map"] = test_map, test_samp
        if val_map > best:
            best, best_epoch, bad, best_test = val_map, epoch, 0, (test_map, test_samp)
            torch.save(state, os.path.join(args.out, "best_weights.pth"))
            pickle.dump(test_probs, open(os.path.join(args.out, "best_test_probs.pkl"), "wb"), pickle.HIGHEST_PROTOCOL)
            msg += "  <- best"
        else:
            bad += 1
        print(msg, flush=True)
        if args.patience and bad >= args.patience:
            print(f"No improvement for {args.patience} epochs -> stopping early.")
            break

    print(f"\nFinished in {(time.time() - t_start) / 60:.1f} min. Best selection mAP: {best:.2f} (epoch {best_epoch})")
    if args.holdout > 0 and best_test:
        print(f"==> FINAL TEST RESULT (model chosen on held-out training videos): "
              f"mAP {best_test[0]:.2f} | sampled mAP {best_test[1]:.2f}")
    print("Saved:", os.path.join(args.out, "best_weights.pth"), "and", log_path)


if __name__ == "__main__":       # required on Windows for DataLoader workers
    main()
