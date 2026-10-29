# Swar-Laya detection backend (FastAPI)

High-quality "Detect Notes" for Tab 2, built around your **line-by-line**
workflow: the vocal is separated once when you open a file, then each Detect
transcribes only that segment — fast on CPU. GPU is config-ready for later.

## Files
- `sargam_mapper_v2.py` — engine (`transcribe`, `detect_segment`, CLI).
- `server.py` — FastAPI app.
- `requirements_backend.txt` — dependencies.

## Install & run
```
pip install -r requirements_backend.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```
Check it: http://127.0.0.1:8000/health   (shows torch version, `cuda`, `demucs`).
First run with Demucs downloads its model (~a few hundred MB) once.

## How the button uses it
1. **Open a file** → `POST /session` (upload once; optional Demucs). Returns a
   `job_id`; poll `GET /job/{id}` until it returns `{session_id, duration}`.
   This is the only place Demucs runs (a few minutes on CPU, once per file).
2. **Each Detect** → `POST /segment` with `session_id, start, beats, bpm, sa,
   model`. Runs CREPE on just that slice and returns per-beat cells. Fast.
3. **Optional pre-analyze** → `POST /analyze` (session_id, model) computes the
   whole-file pitch track in the background with progress, after which every
   `/segment` is instant. Use `model=tiny` here (whole-song `full` is slow).
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
