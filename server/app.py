"""ActionLens - MS-TCT Action Detection as a Service (v6: fused I3D + CLIP, video upload, text search)

Pipeline for an uploaded video (identical to how the training features were made):
  video -> short side 480 px (like Charades_v1_480)
        -> I3D  : 24 fps, centre crop 224, Charades-fine-tuned I3D  -> 1024-D every 8 frames
        -> CLIP : 3 fps, short side 224, centre crop, ViT-B/32       ->  512-D per frame
        -> standardise both (training statistics), resample I3D onto the CLIP time grid
        -> 1536-D x T  ->  MS-TCT (256-snippet windows)  ->  157 action scores per 1/3 s
"""
import io, os, sys, json, time, tempfile, threading, uuid
from collections import OrderedDict
import numpy as np, torch, cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import snapshot_download

sys.path.insert(0, "/opt/MS-TCT")
from MSTCT.MSTCT_Model import MSTCT

# ---------------- model + assets from the Hugging Face model repo ----------------
MODEL_REPO = os.environ.get("MODEL_REPO", "namratha-addagada/ms-tct-charades-fusion")
ASSETS = snapshot_download(MODEL_REPO)
torch.set_num_threads(os.cpu_count() or 2)

ck = torch.load(f"{ASSETS}/mstct_fusion.pth", map_location="cpu", weights_only=True)
IN_DIM, WIN, FPS = int(ck["in_feat_dim"]), int(ck.get("window", 256)), float(ck.get("feature_fps", 3.0))
model = MSTCT([256, 384, 576, 864], 3, 8, 8, IN_DIM, 512, 157)
model.load_state_dict({k: v.float() for k, v in ck["model_state_dict"].items()}, strict=True)
model.eval()

CLASSES = json.load(open(f"{ASSETS}/charades_classes.json"))
SAMPLES = json.load(open(f"{ASSETS}/samples/samples.json"))          # best demo videos first
STATS = np.load(f"{ASSETS}/fusion_stats.npz")
# segment rule tuned on held-out training videos (tune_segments.py); hand-picked fallback
SEG = dict(k=3, rel=0.5, abs=0.05, gap=1.0, min_len=0.6)
# confidence threshold calibrated on held-out videos (calibrate_confidence.py); None = mark everything likely
LIKELY = None
if os.path.exists(f"{ASSETS}/confidence_params.json"):
    LIKELY = float(json.load(open(f"{ASSETS}/confidence_params.json"))["likely_threshold"])
if os.path.exists(f"{ASSETS}/segment_params.json"):
    SEG.update({k: v for k, v in json.load(open(f"{ASSETS}/segment_params.json")).items() if k in SEG})

MAX_UPLOAD = 120 * 1024 * 1024
MAX_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", "30"))
LOCK = threading.Lock()
SESSIONS = OrderedDict()                    # session_id -> (raw CLIP features, duration) of recent uploads                     # one heavy job at a time on the free CPU
_extractors = {}


def extractors():
    """Load I3D + CLIP on first use (keeps start-up fast)."""
    if not _extractors:
        sys.path.insert(0, f"{ASSETS}/i3d")
        from pytorch_i3d import InceptionI3d
        import open_clip
        i3d = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(157)
        i3d.load_state_dict(torch.load(f"{ASSETS}/i3d/rgb_charades.pt", map_location="cpu"))
        clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        _extractors["i3d"], _extractors["clip"] = i3d.eval(), clip.eval()
    return _extractors["i3d"], _extractors["clip"]


# ---------------- video -> features (same rules as extract_*_features.py) ----------------
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def decode(path):
    """Return (frames_24fps_uint8 [T,224,224,3], frames_3fps_float [N,3,224,224], duration)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(400, "Could not read this video. Try an MP4 (H.264) file.")
    native = cap.get(cv2.CAP_PROP_FPS)
    if not native or native < 1 or native > 240:
        native = 30.0
    i3d_f, clip_f, idx, t24, t3, last_t = [], [], 0, 0.0, 0.0, -1.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        # Real timestamp of this frame (browser/phone recordings have a variable frame rate);
        # fall back to frame index / fps when the container gives no timestamps.
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1000.0 if ms and ms / 1000.0 > last_t else idx / native
        if t <= last_t:
            t = last_t + 1.0 / native
        last_t = t
        if t > MAX_SECONDS:
            break
        h, w = fr.shape[:2]
        if min(h, w) != 480:                      # match the 480p Charades videos used in training
            s = 480 / min(h, w)
            fr = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        rgb = None
        while t + 1e-6 >= t24:                    # I3D: true 24 fps (repeat frames if needed)
            if rgb is None:
                H, W = fr.shape[:2]
                y, x = (H - 224) // 2, (W - 224) // 2
                rgb = cv2.cvtColor(fr[y:y + 224, x:x + 224], cv2.COLOR_BGR2RGB)
            i3d_f.append(rgb)
            t24 += 1 / 24
        if t + 1e-6 >= t3:                        # CLIP: 3 fps, short side 224, centre crop
            H, W = fr.shape[:2]
            s = 224 / min(H, W)
            c = cv2.resize(fr, (max(224, round(W * s)), max(224, round(H * s))), interpolation=cv2.INTER_CUBIC)
            H, W = c.shape[:2]
            y, x = (H - 224) // 2, (W - 224) // 2
            c = cv2.cvtColor(c[y:y + 224, x:x + 224], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            clip_f.append(((c - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1))
            t3 += 1 / 3
        idx += 1
    cap.release()
    if len(clip_f) < 3:
        raise HTTPException(400, "Video is too short (need at least 1 second).")
    while len(i3d_f) < 16:
        i3d_f.append(i3d_f[-1])
    return np.stack(i3d_f), np.stack(clip_f), min(max(last_t, len(clip_f) / 3.0), MAX_SECONDS)


def resample(x, n):
    if len(x) == n:
        return x
    src = np.linspace(0, len(x) - 1, n)
    i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (src - i0)[:, None].astype(np.float32)
    return x[i0] * (1 - w) + x[i1] * w


def video_features(path):
    i3d, clip = extractors()
    t0 = time.time()
    f24, f3, dur = decode(path)
    t1 = time.time()
    with torch.inference_mode():
        x = torch.from_numpy(f24).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0
        a = i3d.extract_features(x).squeeze(0).squeeze(-1).squeeze(-1).T.numpy()      # [T/8, 1024]
        t2 = time.time()
        b = torch.cat([clip.encode_image(torch.from_numpy(f3[k:k + 64])) for k in range(0, len(f3), 64)]).numpy()
    t3 = time.time()
    raw_clip = b.copy()
    a = (a - STATS["i3d_mean"]) / STATS["i3d_std"]
    b = (b - STATS["clip_mean"]) / STATS["clip_std"]
    feats = np.concatenate([resample(a, len(b)), b], 1).astype(np.float32)
    timing = {"decode_ms": round((t1 - t0) * 1000), "i3d_ms": round((t2 - t1) * 1000),
              "clip_ms": round((t3 - t2) * 1000)}
    return feats, dur, timing, raw_clip


# ---------------- MS-TCT + reading actions off the scores ----------------
def predict_scores(feats):
    T = len(feats); n = -(-T // WIN)
    x = np.zeros((n * WIN, IN_DIM), np.float32); x[:T] = feats
    x = torch.from_numpy(x).view(n, WIN, IN_DIM).permute(0, 2, 1)
    with torch.inference_mode():
        logits, _ = model(x)
    return torch.sigmoid(logits).reshape(n * WIN, -1)[:T].numpy()


def find_segments(curve, step):
    k = int(SEG["k"])
    c = np.convolve(np.pad(curve, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, "valid") if k > 1 else curve
    on = c >= max(SEG["abs"], SEG["rel"] * c.max())
    segs, start = [], None
    for i, a in enumerate(list(on) + [False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            s, e = start * step, i * step
            if segs and s - segs[-1][1] <= SEG["gap"]:
                segs[-1][1] = e
            else:
                segs.append([s, e])
            start = None
    return [{"start": round(s, 1), "end": round(e, 1)} for s, e in segs if e - s >= SEG["min_len"]]


def curve_points(curve, n=120):
    """Down-sampled confidence curve (0-1) for drawing; keeps responses small."""
    idx = np.linspace(0, len(curve) - 1, min(n, len(curve))).round().astype(int)
    return [round(float(v), 3) for v in curve[idx]]


def summarize(scores, duration, top_k=5, gt_actions=None):
    T = len(scores); step = duration / T
    vs = scores.mean(0)
    top = np.argsort(-vs)[:top_k]
    gt = {int(a[0]) for a in gt_actions} if gt_actions else set()
    out = {"duration": round(duration, 2), "num_snippets": T, "feature_dim": IN_DIM, "likely_threshold": LIKELY, "top_actions": [
        {"class_id": int(c), "name": CLASSES[c], "score": round(float(vs[c]), 4),
         "peak": round(float(scores[:, c].max()), 4), "segments": find_segments(scores[:, c], step),
         "curve": curve_points(scores[:, c]),
         "likely": bool(LIKELY is None or vs[c] >= LIKELY),
         "in_ground_truth": (int(c) in gt) if gt_actions else None} for c in top]}
    if gt_actions:
        out["ground_truth"] = [{"class_id": int(c), "name": CLASSES[int(c)], "start": float(s), "end": float(e)}
                               for c, s, e in gt_actions]
        out["correct_in_top_k"] = sorted(int(c) for c in set(top.tolist()) & gt)
    return out


# ---------------- API ----------------
app = FastAPI(title="MS-TCT Action Detection API",
              description="Temporal action detection (Charades, 157 classes) with fused I3D + CLIP features")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "model_repo": MODEL_REPO, "features": "I3D + CLIP (1536-D)",
            "trained_epoch": ck.get("epoch"), "test_mAP": round(float(ck.get("test_map", 0)), 2),
            "sampled_test_mAP": round(float(ck.get("sampled_test_map", 0)), 2),
            "num_classes": len(CLASSES), "num_samples": len(SAMPLES),
            "max_video_seconds": MAX_SECONDS, "cpu_threads": torch.get_num_threads(), "segment_rule": SEG,
            "likely_threshold": LIKELY}


@app.get("/api/samples")
def samples():
    return [{"video_id": v, "duration": i["duration"],
             "ground_truth": sorted({CLASSES[int(a[0])] for a in i["actions"]})} for v, i in SAMPLES.items()]


class SampleRequest(BaseModel):
    video_id: str
    top_k: int = 5


@app.post("/api/predict/sample")
def predict_sample(req: SampleRequest):
    if req.video_id not in SAMPLES:
        raise HTTPException(404, f"Unknown video_id. Choose from {list(SAMPLES)}")
    t0 = time.time()
    info = SAMPLES[req.video_id]
    feats = np.load(f"{ASSETS}/samples/{req.video_id}.npy").astype(np.float32)
    r = summarize(predict_scores(feats), info["duration"], max(1, min(req.top_k, 20)), info["actions"])
    r.update({"video_id": req.video_id, "latency_ms": round((time.time() - t0) * 1000, 1)})
    return r


@app.post("/api/predict/video")
def predict_video(file: UploadFile = File(...), top_k: int = Form(5)):
    """Upload your own video (MP4/MOV/WebM). Only the first MAX_VIDEO_SECONDS are analysed."""
    data = file.file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Video too large (max 120 MB).")
    t0 = time.time()
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with LOCK, tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data); tmp.close()
        try:
            feats, dur, timing, raw_clip = video_features(tmp.name)
            t1 = time.time()
            r = summarize(predict_scores(feats), dur, max(1, min(top_k, 20)))
            timing["mstct_ms"] = round((time.time() - t1) * 1000)
        finally:
            os.remove(tmp.name)
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = (raw_clip, dur)
    while len(SESSIONS) > 30:
        SESSIONS.popitem(last=False)
    r.update({"session_id": sid, "filename": file.filename, "request_bytes": len(data), "timing": timing,
              "truncated": dur >= MAX_SECONDS - 0.01, "latency_ms": round((time.time() - t0) * 1000, 1)})
    return r


class SearchRequest(BaseModel):
    query: str
    video_id: str | None = None        # a demo video ...
    session_id: str | None = None      # ... or a video uploaded earlier


@app.post("/api/search")
def search(req: SearchRequest):
    """Find the moments that best match a free-text description, using CLIP.
    Works for any phrase, not only the 157 trained actions."""
    q = req.query.strip()[:120]
    if not q:
        raise HTTPException(400, "Type something to search for.")
    if req.video_id in SAMPLES:
        f = np.load(f"{ASSETS}/samples/{req.video_id}.npy").astype(np.float32)[:, 1024:]
        img = f * STATS["clip_std"] + STATS["clip_mean"]          # undo the standardisation -> raw CLIP
        dur = SAMPLES[req.video_id]["duration"]
    elif req.session_id in SESSIONS:
        img, dur = SESSIONS[req.session_id]
    else:
        raise HTTPException(404, "Unknown video. Analyse a video first.")
    t0 = time.time()
    _, clip = extractors()
    import open_clip
    tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    prompts = [q, f"a photo of a person {q}", f"a video frame of someone {q}"]
    with torch.inference_mode():
        txt = clip.encode_text(tok(prompts)).float().numpy()
    txt = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    txt = txt.mean(0); txt /= np.linalg.norm(txt)
    im = img / np.linalg.norm(img, axis=1, keepdims=True)
    sim = im @ txt                                                # cosine similarity per snippet
    sm = np.convolve(np.pad(sim, (1, 1), mode="edge"), np.ones(3) / 3, "valid")
    step = dur / len(sm)
    moments = []
    for i in np.argsort(-sm):                                     # top 3 moments at least 3 s apart
        if all(abs(int(i) - j) * step >= 3 for j, _ in moments):
            moments.append((int(i), float(sm[i])))
        if len(moments) == 3:
            break
    lo, hi = float(sm.min()), float(sm.max())
    return {"query": q, "duration": round(dur, 2),
            "curve": curve_points((sm - lo) / max(hi - lo, 1e-6)),
            "moments": [{"time": round((i + 0.5) * step, 1), "similarity": round(v, 3),
                         "relative": round((v - lo) / max(hi - lo, 1e-6), 2)} for i, v in moments],
            "max_similarity": round(hi, 3), "latency_ms": round((time.time() - t0) * 1000, 1)}


@app.post("/api/predict/features")
async def predict_features(file: UploadFile = File(...), top_k: int = Form(5)):
    """Send pre-computed fused features instead of a video: .npy [T, 1536] at 3 snippets/s."""
    data = await file.read()
    try:
        feats = np.asarray(np.load(io.BytesIO(data), allow_pickle=False), dtype=np.float32)
    except Exception:
        raise HTTPException(400, "Not a valid .npy file.")
    if feats.ndim != 2 or feats.shape[1] != IN_DIM or not 1 <= len(feats) <= 20000:
        raise HTTPException(400, f"Expected shape [T, {IN_DIM}], got {list(feats.shape)}.")
    t0 = time.time()
    r = summarize(predict_scores(feats), len(feats) / FPS, max(1, min(top_k, 20)))
    r.update({"filename": file.filename, "request_bytes": len(data), "latency_ms": round((time.time() - t0) * 1000, 1)})
    return r


# warm up the extractors in the background so the first upload is not slower
threading.Thread(target=extractors, daemon=True).start()

app.mount("/", StaticFiles(directory="web", html=True), name="web")
