r"""
Slide-ready figures + results table from the training logs.

Run from D:\mstct:   python make_plots.py
Output: D:\mstct\results\  (PNG figures at 200 dpi + results_table.csv)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\mstct"
OUT = os.path.join(ROOT, "results")
RUNS = [  # (label, folder, colour)
    ("CLIP (appearance)", "clip_full", "#eda100"),
    ("I3D (motion, paper's features)", "i3d_v2", "#1baf7a"),
    ("I3D + CLIP fusion", "fused", "#2a78d6"),
]
FINAL = ("Final: fusion, held-out selection", "fused_final", "#2a78d6")
INK, MUTED, GRID = "#0b0b0b", "#7a7974", "#e8e7e3"

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100,
})


def read_log(folder):
    path = os.path.join(ROOT, "runs", folder, "log.csv")
    if not os.path.exists(path):
        print("  (missing)", path)
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else None


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", name)


def main():
    os.makedirs(OUT, exist_ok=True)
    logs = {folder: read_log(folder) for _, folder, _ in RUNS + [FINAL]}

    # ---- 1. validation mAP per epoch for the three feature types ----------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, folder, col in RUNS:
        L = logs[folder]
        if not L:
            continue
        ax.plot(L["epoch"], L["val_map"], color=col, lw=2, label=label)
        i = max(range(len(L["val_map"])), key=L["val_map"].__getitem__)
        ax.plot(L["epoch"][i], L["val_map"][i], "o", ms=8, color=col, mec="white", mew=2)
        ax.annotate(f"{L['val_map'][i]:.2f}", (L["epoch"][i], L["val_map"][i]), xytext=(6, 6),
                    textcoords="offset points", color=INK, fontsize=11)
    ax.set(xlabel="Epoch", ylabel="Test mAP (%)", ylim=(0, None),
           title="MS-TCT on the full Charades set: which features help?")
    ax.legend(loc="lower right")
    save(fig, "1_feature_comparison_curves.png")

    # ---- 2. final run: losses (overfitting) + mAP, chosen epoch marked -----------
    L = logs["fused_final"]
    if L:
        best = max(range(len(L["val_map"])), key=L["val_map"].__getitem__)
        ep = L["epoch"][best]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
        a1.plot(L["epoch"], L["train_loss"], color="#6250d6", lw=2, label="Training loss")
        a1.plot(L["epoch"], L["val_loss"], color="#eb6834", lw=2, label="Held-out loss")
        a1.set(xlabel="Epoch", ylabel="Loss", title="Losses: the model starts to memorise")
        a1.legend()
        a2.plot(L["epoch"], L["val_map"], color="#9a9893", lw=2, label="Held-out training videos (selection)")
        a2.plot(L["epoch"], L["test_map"], color="#2a78d6", lw=2, label="Official test set")
        a2.set(xlabel="Epoch", ylabel="mAP (%)", ylim=(0, None), title="mAP: epoch chosen on held-out videos only")
        a2.legend(loc="lower right")
        for a in (a1, a2):
            a.axvline(ep, color=INK, lw=1, ls="--")
        a2.annotate(f"chosen: epoch {ep:.0f}\ntest mAP {L['test_map'][best]:.2f}",
                    (ep, L["test_map"][best]), xytext=(12, -42), textcoords="offset points", color=INK,
                    fontsize=11, arrowprops=dict(arrowstyle="-", color=MUTED))
        save(fig, "2_final_run_losses_and_map.png")

    # ---- 3. the journey: one bar per stage ------------------------------------
    stages = [("Colab baseline\n93 videos", 4.84, "#c3c2b7"),
              ("Colab\n1,500 videos", 10.90, "#c3c2b7")]
    for label, folder, col in RUNS:
        if logs[folder]:
            short = {"clip_full": "CLIP\n(appearance)", "i3d_v2": "I3D\n(motion)", "fused": "I3D + CLIP\nfusion"}[folder]
            stages.append((short, max(logs[folder]["val_map"]), col))
    if L:
        stages.append(("Final (fusion)\nheld-out selection", L["test_map"][best], "#1c5cab"))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    bars = ax.bar(range(len(stages)), [s[1] for s in stages], color=[s[2] for s in stages],
                  width=0.62, edgecolor="white", linewidth=2)
    for b, (_, v, _) in zip(bars, stages):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}", ha="center", color=INK, fontsize=12)
    ax.axhline(25.4, color=MUTED, lw=1, ls=":")
    ax.text(len(stages) - 0.5, 25.8, "paper (I3D) ~25", ha="right", color=MUTED, fontsize=10)
    ax.set_xticks(range(len(stages)), [s[0] for s in stages], fontsize=10)
    ax.set(ylabel="Per-frame mAP (%)", ylim=(0, 29), title=f"From {stages[0][1]:.1f}% to {stages[-1][1]:.1f}% mAP")
    ax.text(0.5, -0.22, "Grey: early Colab runs (VGG-16 features, scored on a 300-video test subset). "
            "Colour: full training set, all 1,814 official test videos.", transform=ax.transAxes,
            ha="center", va="top", color=MUTED, fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "3_progress_bar_chart.png")

    # ---- 4. results table ----------------------------------------------------
    with open(os.path.join(OUT, "results_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "features", "best_epoch", "test_map", "sampled_test_map", "selection"])
        for label, folder, _ in RUNS:
            R = logs[folder]
            if R:
                i = max(range(len(R["val_map"])), key=R["val_map"].__getitem__)
                w.writerow([folder, label, int(R["epoch"][i]), R["val_map"][i], R["sampled_val_map"][i], "test set (official code)"])
        if L:
            w.writerow(["fused_final", FINAL[0], int(ep), L["test_map"][best], L["sampled_test_map"][best],
                        "held-out 10% of training videos"])
    print("  saved results_table.csv")
    print("\nAll figures in", OUT)


if __name__ == "__main__":
    main()
