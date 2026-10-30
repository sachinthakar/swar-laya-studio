"""
model_engine — our trained CRNN model (indian_transcriber_v2_best.h5) as a
swara-class engine for server.py, alongside the existing CREPE engine in
sargam_mapper_v2.py.

Why this is a separate file, not a mode inside sargam_mapper_v2.py: the two
engines produce a frame-level class array in different ways and at different
frame rates, but both feed the SAME downstream beat-slicing code
(beat_subnotes / beat_cell in sargam_mapper_v2.py) once they're expressed as
an `idx` array -- idx[i] = octave*12 + swara_class (0-11), or None for a rest
-- exactly what sargam_mapper_v2.frame_swaras() already returns for CREPE.
That reuse is the point: this module's only job is to produce an `idx` array
and the frame rate it's sampled at; everything after that (sub-note
splitting within a beat, Sam/Khali, sustain) is untouched, shared code.

Architecture, CQT geometry and class table are copied verbatim from
transcribe_song_studio_v2.py (the offline CLI) so predictions made here
match what that script would produce on the same audio -- these constants
MUST match training (train_indian_transcriber_kaggle.py) exactly, so don't
"clean them up" independently of that file.

Octave is NOT part of the model's output (it predicts swara class only,
0-11, or 12 for rest). The offline CLI derives octave from a separate, slow,
resumable pyin pass designed for a batch script that can be re-run several
times on a long file. That doesn't fit a single interactive "Pre-analyze"
HTTP call, so here octave instead comes from CREPE's pitch track (already a
fast, available dependency via sargam_mapper_v2.crepe_voiced_only /
crepe_pitch) -- a deliberate adaptation, not an oversight. If the offline
CLI's exact pyin-based octave logic is ever wanted here too, swap out
`octave_from_crepe()` for a port of transcribe_song_studio_v2.estimate_octaves
plus estimate_f0_resumable; nothing else in this file would need to change.
"""

import math
import os

import numpy as np
import librosa

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "indian_transcriber_v2_best.h5")

# ── CQT geometry + class table (must match train_indian_transcriber_kaggle.py) ─
SAMPLE_RATE = 22050
HOP_LENGTH = 512
CHUNK_DURATION = 10.0
BINS_PER_OCTAVE = 36
N_OCTAVES_SPAN = 5
N_BINS = BINS_PER_OCTAVE * N_OCTAVES_SPAN        # 180
FMIN = librosa.note_to_hz("C1")
TONIC_REF_BIN = int(round(BINS_PER_OCTAVE * 2.5))
NUM_MELODY_CLASSES = 13                           # 0-11 swara, 12 = rest
MODEL_FPS = SAMPLE_RATE / HOP_LENGTH              # ~43.07 fps

_model_singleton = {"model": None}


def model_available():
    return os.path.exists(_MODEL_PATH)


def _tonic_shift_bins(tonic_hz):
    raw_tonic_bin = BINS_PER_OCTAVE * np.log2(tonic_hz / FMIN)
    return int(round(TONIC_REF_BIN - raw_tonic_bin))


def _extract_tonic_normalized_cqt(audio, sr, tonic_hz):
    cqt = np.abs(librosa.cqt(audio, sr=sr, hop_length=HOP_LENGTH,
                              fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    frames = cqt_db.T
    shift = _tonic_shift_bins(tonic_hz)
    shifted = np.roll(frames, shift, axis=1)
    if shift > 0:
        shifted[:, :shift] = cqt_db.min()
    elif shift < 0:
        shifted[:, shift:] = cqt_db.min()
    return shifted


def _build_model(input_shape):
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Reshape,
                                          Bidirectional, LSTM, Dense, Dropout)
    from tensorflow.keras.models import Model
    inputs = Input(shape=input_shape, name="audio_cqt_tonic_normalized")
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((1, 3))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((1, 3))(x)
    x = Dropout(0.3)(x)
    x = Reshape((-1, x.shape[2] * x.shape[3]))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    melody_out = Dense(NUM_MELODY_CLASSES, activation='softmax', name="melody")(x)
    return Model(inputs=inputs, outputs=melody_out)


def get_model():
    """Lazy singleton -- the model loads once per server process, on first use
    (not at import time), so the backend still starts fast when the model
    engine is never touched in a session."""
    if _model_singleton["model"] is not None:
        return _model_singleton["model"]
    import tensorflow as tf
    if not model_available():
        raise RuntimeError(
            f"Model file not found at {_MODEL_PATH}. "
            f"Copy indian_transcriber_v2_best.h5 into the models/ folder.")
    dummy = np.zeros(int(CHUNK_DURATION * SAMPLE_RATE), dtype=np.float32)
    input_shape = _extract_tonic_normalized_cqt(dummy, SAMPLE_RATE, 220.0)[..., np.newaxis].shape
    try:
        model = tf.keras.models.load_model(_MODEL_PATH)
    except Exception:
        model = _build_model(input_shape)
        model.load_weights(_MODEL_PATH)
    _model_singleton["model"] = model
    return model


def run_model_inference(y_22050, tonic_hz, progress=None):
    """y_22050: mono float audio AT SAMPLE_RATE (22050 Hz) -- resample before
    calling this if the source audio is at a different rate. Same fixed-10s-
    window chunking as training/the offline CLI. Returns melody_classes, an
    int array (0-11 swara, 12 rest) at MODEL_FPS."""
    model = get_model()
    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    y = y_22050
    remainder = len(y) % chunk_samples
    if remainder != 0:
        y = np.pad(y, (0, chunk_samples - remainder))
    num_chunks = len(y) // chunk_samples
    if num_chunks == 0:
        return np.zeros(0, dtype=int)

    X_batch = []
    for i in range(num_chunks):
        chunk_y = y[i * chunk_samples:(i + 1) * chunk_samples]
        cqt_frames = _extract_tonic_normalized_cqt(chunk_y, SAMPLE_RATE, tonic_hz)
        X_batch.append(cqt_frames[..., np.newaxis])
        if progress:
            progress((i + 1) / num_chunks)
    X_batch = np.array(X_batch)

    predictions = model.predict(X_batch, verbose=0)
    return np.argmax(predictions, axis=-1).flatten()


def octave_from_crepe(melody_classes, crepe_f0, crepe_pd, crepe_fps, sa_hz, conf=0.5):
    """Per model-frame octave, looked up from CREPE's pitch track (fast
    alternative to the offline CLI's slow resumable-pyin octave pass -- see
    the module docstring). Nearest-frame lookup; f0<=0 or low-confidence
    frames default to octave 0 (harmless -- such frames are usually rest
    anyway, where octave is unused)."""
    n = len(melody_classes)
    octaves = np.zeros(n, dtype=int)
    for i in range(n):
        t = i / MODEL_FPS
        j = int(round(t * crepe_fps))
        if 0 <= j < len(crepe_f0) and crepe_pd[j] > conf and crepe_f0[j] > 0:
            semitones = 12 * math.log2(crepe_f0[j] / sa_hz)
            octaves[i] = int(np.floor(round(semitones) / 12))
    return octaves


def classes_to_idx(melody_classes, octaves):
    """(melody_classes, octaves) -> idx array in the same encoding
    sargam_mapper_v2.frame_swaras() produces: idx[i] = octave*12 + class
    (0-11), or None for a rest (class 12). This is the bridge into the
    existing, shared beat_subnotes()/beat_cell() beat-slicing code."""
    idx = np.full(len(melody_classes), None, dtype=object)
    for i, c in enumerate(melody_classes):
        if c == 12:
            continue
        idx[i] = int(octaves[i]) * 12 + int(c)
    return idx


def idx_from_model(y_22050, tonic_hz, crepe_f0, crepe_pd, crepe_fps, progress=None):
    """Full bridge: audio (+ a CREPE pitch track for octave) -> idx array at
    MODEL_FPS, ready for sargam_mapper_v2.beat_subnotes(). Used by server.py
    (the local backend, which already has torch/torchcrepe for the CREPE
    engine, so reusing its pitch track for octave costs nothing extra)."""
    melody_classes = run_model_inference(y_22050, tonic_hz, progress=progress)
    octaves = octave_from_crepe(melody_classes, crepe_f0, crepe_pd, crepe_fps, tonic_hz)
    return classes_to_idx(melody_classes, octaves), MODEL_FPS


def octave_from_pyin(melody_classes, y_22050, sa_hz, conf=0.5):
    """Octave source for a torch-free deployment (see server_model.py): same
    job as octave_from_crepe(), but via librosa.pyin instead of CREPE, so this
    module's only runtime dependencies stay TensorFlow + librosa -- no torch,
    torchcrepe, or Demucs. Pitch accuracy is a notch below CREPE, but octave
    only needs to land in the right 12-semitone bracket, which pyin does
    reliably enough on a monophonic-ish melody line.

    pyin is run with the SAME sr/hop_length as the model's CQT front end, so
    its frames line up ~1:1 with `melody_classes` by index (no resampling, no
    time lookup -- unlike octave_from_crepe, which must map between two
    different frame rates)."""
    import librosa
    fmin, fmax = max(65.0, sa_hz * 0.4), sa_hz * 3.2
    f0, _voiced_flag, voiced_prob = librosa.pyin(
        y_22050, sr=SAMPLE_RATE, fmin=fmin, fmax=fmax,
        frame_length=2048, hop_length=HOP_LENGTH, fill_na=0.0)
    n = len(melody_classes)
    octaves = np.zeros(n, dtype=int)
    for i in range(min(n, len(f0))):
        if voiced_prob[i] > conf and f0[i] > 0:
            semitones = 12 * math.log2(f0[i] / sa_hz)
            octaves[i] = int(np.floor(round(semitones) / 12))
    return octaves


def idx_from_model_standalone(y_22050, tonic_hz, progress=None):
    """Self-contained bridge for server_model.py: no external pitch track
    needed, just audio + tonic. This is what makes the trimmed Render
    deployment possible -- everything it touches (this function, the model,
    librosa.pyin) needs only TensorFlow + librosa, never torch/Demucs."""
    melody_classes = run_model_inference(y_22050, tonic_hz, progress=progress)
    octaves = octave_from_pyin(melody_classes, y_22050, tonic_hz)
    return classes_to_idx(melody_classes, octaves), MODEL_FPS


def beats_for_window_idx(idx, fps, start_sec, n_beats, bpm):
    """Same shape/contract as sargam_mapper_v2.beats_for_window(), but takes a
    pre-built idx array (from this module) instead of computing one from
    (f0, pd) via frame_swaras(). voiced_pct is approximated as "not a rest"
    rather than a CREPE confidence, since the model has no confidence score."""
    import sargam_mapper_v2 as sm
    beat = 60.0 / bpm
    out = []
    for i in range(n_beats):
        t0 = start_sec + i * beat
        f_lo = int(round(t0 * fps)); f_hi = int(round((t0 + beat) * fps))
        window = idx[f_lo:min(f_hi, len(idx))]
        voiced_pct = int(100 * np.mean([x is not None for x in window])) if len(window) else 0
        sub = sm.beat_subnotes(idx, fps, t0, beat)
        n, o = sm.beat_cell(sub)
        out.append({"notes": n, "octs": o, "voiced_pct": voiced_pct})
    return out
