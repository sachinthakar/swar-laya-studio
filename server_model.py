"""
server_model.py — standalone "Our Trained Model" backend, deployable on Render
================================================================================
Same API shape as server.py's model-engine path, but with NO torch,
torchcrepe, or Demucs dependency — only FastAPI + TensorFlow + librosa/scipy
(see requirements_render.txt). That's what makes this small enough to run as
a normal paid Render Web Service instead of needing a local machine.

Trade-off accepted for that: octave comes from librosa.pyin (model_engine.
octave_from_pyin) instead of CREPE, and there is no vocal-separation step —
both match how the offline CLI (transcribe_song_studio_v2.py) actually runs
the trained model: on the full mix, with pyin for octave. The CREPE engine
and Demucs vocal separation remain local-only (server.py); this service does
not attempt them.

Run locally the same way:
    pip install -r requirements_render.txt
    uvicorn server_model:app --host 127.0.0.1 --port 8001

On Render: see render.yaml (service name swar-laya-model) and the "Deploying
the trained-model engine on Render" section of README_backend.md.

Endpoints (same shapes the Studio's SMB client already speaks, for
engine=model requests, so public/v5/transcribe.js needs no code change --
only the "Backend URL" field has to point at this service instead of
127.0.0.1:8000 when the user wants the trained model without running
anything locally):
    GET  /health
    GET  /job/{id}
    POST /session   (file)                                  -> job_id
    POST /segment   (session_id, start, beats, bpm, sa)      -> beats[]
    POST /analyze   (session_id)                             -> job_id
"""

import os, uuid, tempfile, threading, traceback, math
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sargam_mapper_v2 as eng   # only parse_sa/detect_sa/beat_subnotes/beat_cell are
                                  # used below -- none of those need torch/Demucs (see
                                  # model_engine.py's module docstring for why that's safe).
import model_engine as meng

app = FastAPI(title="Swar-Laya Trained-Model Backend", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

SR = meng.SAMPLE_RATE              # 22050 -- the model's native rate; loaded straight at this rate
_LOCK = threading.Lock()           # serialize model inference (TF session isn't thread-safe by default)
SESSIONS = {}                      # sid -> {y, duration, sa, idx, fps}
JOBS = {}


def _run_job(fn):
    jid = uuid.uuid4().hex
    JOBS[jid] = {"status": "running", "progress": 0.0, "message": "", "result": None, "error": None}
    def worker():
        try:
            JOBS[jid]["result"] = fn(jid)
            JOBS[jid]["status"] = "done"; JOBS[jid]["progress"] = 1.0
        except Exception as e:
            traceback.print_exc()
            JOBS[jid]["status"] = "error"; JOBS[jid]["error"] = f"{type(e).__name__}: {e}"
    threading.Thread(target=worker, daemon=True).start()
    return jid

def _set(jid, **kw):
    if jid in JOBS:
        JOBS[jid].update(kw)


@app.get("/health")
def health():
    info = {"status": "ok", "sessions": len(SESSIONS), "model_engine": meng.model_available()}
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
    except Exception:
        info["tensorflow"] = None
    return info


@app.get("/job/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    return j


# ── 1. open a file ───────────────────────────────────────────────────────────
@app.post("/session")
def session(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "a.mp3")[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read()); tmp.flush(); tmp.close()

    def work(jid):
        _set(jid, message="loading audio…")
        y, _sr = librosa.load(tmp.name, sr=SR, mono=True)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {"y": y, "duration": len(y) / SR, "sa": None, "idx": None, "fps": None}
        return {"session_id": sid, "duration": len(y) / SR, "demucs_used": False}

    return {"job_id": _run_job(work)}


def _resolve_sa(sess, sa):
    if sa and sa != "auto":
        return eng.parse_sa(sa)
    if sess.get("sa"):
        return sess["sa"]
    clip = sess["y"][: int(60 * SR)]
    f0, _flag, prob = librosa.pyin(clip, sr=SR, fmin=70, fmax=600,
                                   frame_length=2048, hop_length=meng.HOP_LENGTH, fill_na=0.0)
    sa_hz = eng.detect_sa(f0, prob > 0.5)
    sess["sa"] = sa_hz or 220.0
    return sess["sa"]


def _segment_uncached(sess, sa_hz, start, beats, bpm):
    beat_dur = 60.0 / bpm
    seg_start = max(0.0, start)
    seg_len = max(beats * beat_dur, meng.CHUNK_DURATION)
    s = int(seg_start * SR); e = int((seg_start + seg_len) * SR)
    clip = sess["y"][s:e]
    need = int(meng.CHUNK_DURATION * SR)
    if len(clip) < need:
        clip = np.pad(clip, (0, need - len(clip)))
    with _LOCK:
        idx, fps = meng.idx_from_model_standalone(clip, sa_hz)
    return meng.beats_for_window_idx(idx, fps, start - seg_start, beats, bpm)


# ── 2. per-line detect ───────────────────────────────────────────────────────
@app.post("/segment")
def segment(session_id: str = Form(...), start: float = Form(...),
            beats: int = Form(...), bpm: float = Form(...),
            sa: str = Form("auto"), model: str = Form("full"), device: str = Form("cpu")):
    # `model`/`device` are accepted and ignored -- the Studio's SMB client
    # always sends them; they're CREPE-only/local-backend-only settings.
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "unknown session (re-open the file)")
    if not meng.model_available():
        raise HTTPException(500, "model file missing on server "
                                  "(models/indian_transcriber_v2_best.h5)")
    sa_hz = _resolve_sa(sess, sa)
    if sess.get("idx") is not None:
        cells = meng.beats_for_window_idx(sess["idx"], sess["fps"], start, beats, bpm)
    else:
        cells = _segment_uncached(sess, sa_hz, start, beats, bpm)
    return {"beats": cells, "sa": round(sa_hz, 2)}


# ── 3. optional: pre-compute whole-file predictions ─────────────────────────
@app.post("/analyze")
def analyze(session_id: str = Form(...), model: str = Form("full"), device: str = Form("cpu")):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "unknown session")
    if not meng.model_available():
        raise HTTPException(500, "model file missing on server "
                                  "(models/indian_transcriber_v2_best.h5)")

    def work(jid):
        sa_hz = _resolve_sa(sess, "auto")
        _set(jid, message="running trained model over whole file…")
        with _LOCK:
            idx, fps = meng.idx_from_model_standalone(
                sess["y"], sa_hz, progress=lambda p: _set(jid, progress=p))
        sess["idx"], sess["fps"] = idx, fps
        return {"session_id": session_id, "frames": int(len(idx))}

    return {"job_id": _run_job(work)}
