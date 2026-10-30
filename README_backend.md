# Swar-Laya detection backend (FastAPI)

High-quality "Detect Notes" for Tab 2, built around your **line-by-line**
workflow: the vocal is separated once when you open a file, then each Detect
transcribes only that segment — fast on CPU. GPU is config-ready for later.

## Files
- `sargam_mapper_v2.py` — CREPE engine (`transcribe`, `detect_segment`, CLI).
- `model_engine.py` — our trained CRNN as an alternate engine (see "Two engines" below).
- `models/indian_transcriber_v2_best.h5` — the trained model weights `model_engine.py` loads.
- `server.py` — FastAPI app.
- `requirements_backend.txt` — dependencies.

## Install & run
```
pip install -r requirements_backend.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```
Check it: http://127.0.0.1:8000/health   (shows torch version, `cuda`, `demucs`).
First run with Demucs downloads its model (~a few hundred MB) once.

To Run trained model at the same port, and use it locally: 

uvicorn server_model:app --host 127.0.0.1 --port 8000

So in practice: run server.py locally for either engine, 
or run server_model.py locally for a lighter trained-model-only setup 
— whichever is running at the time determines which Engine selection actually works, 
since the frontend has no way to know which one is live.

## How the button uses it
1. **Open a file** → `POST /session` (upload once; optional Demucs). Returns a
   `job_id`; poll `GET /job/{id}` until it returns `{session_id, duration}`.
   This is the only place Demucs runs (a few minutes on CPU, once per file).
2. **Each Detect** → `POST /segment` with `session_id, start, beats, bpm, sa,
   model, engine`. Runs the selected engine on just that slice and returns
   per-beat cells. Fast.
3. **Optional pre-analyze** → `POST /analyze` (session_id, model, engine)
   computes the whole-file pitch/class track in the background with progress,
   after which every `/segment` is instant. Use `model=tiny` here when
   `engine=crepe` (whole-song `full` is slow); `model` is ignored for
   `engine=model`.
4. **Whole song** → `POST /detect` (file + params) runs the batch pipeline
   (vocal-only speedup) and returns the full Studio JSON. For CLI/batch.

`/segment` returns:
```json
{ "sa": 207.65,
  "beats": [ {"notes":["dha","Pa"], "octs":[-1,-1], "voiced_pct":90}, ... ] }
```
`notes`/`octs` are per beat with 1–4 sub-parts (off-beat notes split the beat),
`-` = rest. Feed straight into the Studio's existing display/accept/edit flow;
its `smApplySustain` adds the `~` holds across beats.

## Two engines
Both `/segment` and `/analyze` take an `engine` field:
- `engine=crepe` (default) — pitch-tracking via `sargam_mapper_v2.py`. No
  extra install beyond `requirements_backend.txt`'s base dependencies.
- `engine=model` — our trained CRNN (`model_engine.py` +
  `models/indian_transcriber_v2_best.h5`), the same model the offline
  `transcribe_song_studio_v2.py` CLI uses (in the separate Indian Music
  Transcription project). Needs `tensorflow` installed (in
  `requirements_backend.txt`) and the model file present. The model predicts
  swara class only (not octave), so octave is filled in from a CREPE pitch
  pass over the same audio — a deliberate choice for interactive use (the
  offline CLI's own octave step is a slow, resumable pyin pass meant to be
  re-run several times on one file, which doesn't fit a single HTTP call).
  `model` (full/tiny) is a CREPE-only setting and has no effect on this engine.
  `/health` reports `model_engine` (file present) and `tensorflow` (installed)
  so the Studio UI can tell whether this engine is actually usable.

## Speed & quality knobs
- `model`: `full` (best; ~25 s per 6-beat segment on CPU) or `tiny`
  (~1 s per segment, lower accuracy — good for a first pass).
- Pre-analyze with `tiny` (~1.5 min for a 5-min song) → later segments instant.
- `demucs=true` at `/session` cleans harmonium bleed from the voice (recommended).
- **GPU (future):** everything takes a `device` field — `cpu` (default), `cuda`,
  or `auto` (GPU when present). Install a CUDA build of torch and pass
  `device=auto`; `/health` will show `"cuda": true`. Full-model segments then
  drop to ~1 s and whole-song analyze to a couple of minutes.
- Whole-file `/detect` uses a **vocal-only** pass (skips silent/instrumental
  gaps) by default; pass `whole_song=true` to process every frame.

## Deploying the trained-model engine on Render (optional)
To use "Our Trained Model" without running anything locally, `server_model.py`
is a second, slimmer backend that can run as its own Render Web Service.

What's different from `server.py`:
- Only the trained model + `librosa.pyin` for octave (no CREPE/torch,
  no Demucs vocal separation) -- so the install is small enough for a normal
  paid Render plan. This matches how the offline CLI
  (`transcribe_song_studio_v2.py`) actually runs the model: on the full mix,
  pyin for octave, no vocal isolation.
- The CREPE engine and vocal separation stay local-only; this service does
  not replace `server.py`, it only takes over the trained-model engine.

Files it needs: `server_model.py`, `model_engine.py`,
`models/indian_transcriber_v2_best.h5`, `sargam_mapper_v2.py` (only its
`parse_sa`/`detect_sa`/`beat_subnotes`/`beat_cell` helpers are used --
none of those import torch/Demucs, so having the file present is enough,
no extra install), `requirements_render.txt`.

To deploy:
1. Render Dashboard -> New -> Blueprint -> this repo, `render-deploy` branch.
   It will offer `swar-laya-model` (defined in `render.yaml`) alongside the
   existing static site. Pick an instance plan with enough RAM for
   TensorFlow + librosa loaded together -- check Render's current plan specs
   first; the free tier is unlikely to be enough and will sleep when idle.
2. Once it's live, copy its URL (e.g. `https://swar-laya-model.onrender.com`).
3. In the Studio's Transcribe-from-Audio panel: tick "AI backend", paste that
   URL into "Backend URL" (replacing `http://127.0.0.1:8000`), and pick
   "Engine: Our Trained Model". No local server needed for that engine from
   then on; "Engine: CREPE" still needs `server.py` running locally.

Endpoints: same shapes as `server.py`'s model-engine path --
`GET /health` · `GET /job/{id}` · `POST /session` · `POST /segment` ·
`POST /analyze` (no `/detect` batch endpoint on this service).

## CLI (no server)
```
python sargam_mapper_v2.py "Lajun Hasane.mp3" --sa G#3 --bpm 140 --beats 6 \
    --sam 2.38 --vocal-start 10 --demucs --model full --wav
#   add --device auto  to use a GPU when available
#   add --whole-song   to disable the vocal-only speedup
```
Writes `<song>_swarlekh.json` (load in the Studio) and, with `--wav`, a resynth.

## Endpoints
`GET /health` · `GET /job/{id}` · `POST /session` · `POST /segment` ·
`POST /analyze` · `POST /detect`
