"""
Swar-Laya detection backend — FastAPI
=====================================
Matches the Studio's LINE-BY-LINE workflow: the vocal is separated once when a
file is opened, then each "Detect Notes" click transcribes just that segment —
fast on CPU. GPU is config-ready (device="auto"/"cuda") for the future.

Run:
    pip install -r requirements_backend.txt
    uvicorn server:app --host 127.0.0.1 --port 8000
    # open http://127.0.0.1:8000/health

Flow used by the button
-----------------------
 1. On file load:  POST /session  (upload once; optional Demucs) -> job_id
                   GET  /job/{id} -> {session_id, duration}          (poll progress)
 2. Each Detect:   POST /segment  (session_id, start, beats, bpm, sa) -> beats[]
                   (fast; runs CREPE on just that slice, or a cached track)
 3. Optional:      POST /analyze  (session_id) -> job_id             (pre-compute the
                   whole-file pitch track with progress; makes every later
                   /segment instant)
 4. Whole song:    POST /detect   (file, params) -> job_id -> full Studio JSON
                   (vocal-only speedup; for batch/CLI parity)

Config
------
DEVICE default is "cpu". Set the `device` field to "cuda" or "auto" once a
CUDA build of torch is installed to run on GPU (much faster).
"""

import os, io, uuid, tempfile, threading, traceback
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sargam_mapper_v2 as eng

app = FastAPI(title="Swar-Laya Detection Backend", version="2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

SR, HOP = 16000, 160
_LOCK = threading.Lock()           # serialize heavy torch work
SESSIONS = {}                      # sid -> {vocal, duration, demucs_used, f0, pd, sa}
JOBS = {}                          # jid -> {status, progress, message, result, error}


# ── background-job helper ────────────────────────────────────────────────────
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
    info = {"status": "ok", "sessions": len(SESSIONS), "demucs": False}
    try:
        import demucs; info["demucs"] = True            # noqa
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except Exception:
        info["torch"] = None
    return info


@app.get("/job/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    return j


# ── 1. open a file: isolate vocal once ───────────────────────────────────────
@app.post("/session")
def session(file: UploadFile = File(...), demucs: bool = Form(False),
            device: str = Form("cpu")):
    suffix = os.path.splitext(file.filename or "a.mp3")[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read()); tmp.flush(); tmp.close()

    def work(jid):
        _set(jid, message="separating vocal…")
        with _LOCK:
            vocal, used = eng.isolate_vocal(tmp.name, SR, demucs, log=lambda m: _set(jid, message=str(m)),
                                            device=device)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {"vocal": vocal, "duration": len(vocal)/SR,
                         "demucs_used": used, "f0": None, "pd": None, "sa": None}
        return {"session_id": sid, "duration": len(vocal)/SR, "demucs_used": used}

    return {"job_id": _run_job(work)}


def _resolve_sa(sess, sa, model, device):
    if sa and sa != "auto":
        return eng.parse_sa(sa)
    if sess.get("sa"):
        return sess["sa"]
    # detect from cached track, else a quick tiny pass over <=60s of vocal
    if sess["f0"] is not None:
        sa_hz = eng.detect_sa(sess["f0"], sess["pd"] > 0.5)
    else:
        clip = sess["vocal"][: int(60 * SR)]
        with _LOCK:
            f0, pd = eng.crepe_pitch(clip, SR, HOP, 90, 600, "tiny", device)
        sa_hz = eng.detect_sa(f0, pd > 0.5)
    sess["sa"] = sa_hz or 220.0
    return sess["sa"]


# ── 2. line-by-line detect (fast) ────────────────────────────────────────────
@app.post("/segment")
def segment(session_id: str = Form(...), start: float = Form(...),
            beats: int = Form(...), bpm: float = Form(...),
            sa: str = Form("auto"), model: str = Form("full"),
            device: str = Form("cpu")):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "unknown session (re-open the file)")
    sa_hz = _resolve_sa(sess, sa, model, device)
    with _LOCK:
        cells = eng.detect_segment(sess["vocal"], SR, sa_hz, start, beats, bpm,
                                   model=model, device=device, hop=HOP,
                                   f0_cache=sess["f0"], pd_cache=sess["pd"])
    return {"beats": cells, "sa": round(sa_hz, 2)}


# ── 3. optional: pre-compute whole-file pitch so segments become instant ─────
@app.post("/analyze")
def analyze(session_id: str = Form(...), model: str = Form("full"),
            device: str = Form("cpu")):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "unknown session")

    def work(jid):
        sa_hz = _resolve_sa(sess, "auto", model, device)
        fmin, fmax = max(70, sa_hz*0.4), sa_hz*3.2
        _set(jid, message="tracking pitch over whole file…")
        with _LOCK:
            f0, pd = eng.crepe_voiced_only(sess["vocal"], SR, HOP, fmin, fmax,
                                           model, device, progress=lambda p: _set(jid, progress=p))
        sess["f0"], sess["pd"] = f0, pd
        return {"session_id": session_id, "frames": int(len(f0))}

    return {"job_id": _run_job(work)}


# ── 4. whole-song batch (vocal-only speedup) ─────────────────────────────────
@app.post("/detect")
def detect(file: UploadFile = File(...), bpm: float = Form(...),
           beats: int = Form(6), sa: str = Form("auto"), sam: str = Form("auto"),
           vocal_start: str = Form("auto"), demucs: bool = Form(False),
           model: str = Form("full"), device: str = Form("cpu"),
           whole_song: bool = Form(False), max_seconds: float = Form(0),
           title: str = Form("")):
    suffix = os.path.splitext(file.filename or "a.mp3")[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read()); tmp.flush(); tmp.close()

    def work(jid):
        logs = []
        with _LOCK:
            doc = eng.transcribe(tmp.name, sa=sa, bpm=bpm, beats=beats, sam=sam,
                                 vocal_start=vocal_start, demucs=demucs, model=model,
                                 device=device, vocal_only=not whole_song,
                                 max_seconds=(max_seconds or None),
                                 title=(title or (file.filename or "Transcription")),
                                 log=lambda m: logs.append(str(m)),
                                 progress=lambda p: _set(jid, progress=p))
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        doc.setdefault("_meta", {})["log"] = logs
        return doc

    return {"job_id": _run_job(work)}


# ── Serve the Studio app + samples from ./public ─────────────────────────────
# So one server does everything: open  http://127.0.0.1:8000/v4.html
#   - the app and its /samples/... load from public/ (fixes sample playback),
#   - and it's the SAME origin as the API above (no CORS, no mixed content).
# Mounted LAST so the API routes above take precedence.
try:
    from fastapi.staticfiles import StaticFiles
    if os.path.isdir("public"):
        app.mount("/", StaticFiles(directory="public", html=True), name="app")
        print("Serving ./public at http://127.0.0.1:8000/  (open /v4.html)")
    else:
        print("No ./public folder next to server.py — static app not served "
              "(run uvicorn from the project root).")
except Exception as _e:
    print("Static mount skipped:", _e)
