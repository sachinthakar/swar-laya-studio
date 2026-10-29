"""
Sargam Mapper — Swar-Laya Studio
Transcribes a vocal/melodic audio file into the Swar-Laya JSON format.

Key improvements over the previous version
------------------------------------------
- librosa.pyin pitch detection (much more accurate than custom YIN)
- HPSS harmonic separation + vocal-range bandpass filter before pyin
- Full pitch track pre-computed; real-time loop only does playback + keyboard
- Octave-error correction: doubles frames below SA_HZ/2 (pyin octave artifacts)
- Median-filter vibrato smoothing on the pitch track
- Frame-accurate beat aggregation using pyin frame indices (no chunk drift)
- SCALE_MAP updated to match the new Swar-Laya scale selector (actual Sa Hz)
- AI prompt includes song/raga metadata and slow-song-specific rules
"""

import numpy as np
import json
import math
import sys
import librosa
from scipy.signal import butter, lfilter, find_peaks
import scipy.ndimage
import requests
import os
from collections import Counter

# ── API keys ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyA400Elj7amr8zq4uBC4Xw3CJGNl2irnm0"
MISTRAL_API_KEY = "WaeawLQ7ysDSf0nCae1pBhE11AcPhkpg"

# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────
def call_gemini(prompt):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-pro:generateContent?key={GEMINI_API_KEY}")
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192,
                             "responseMimeType": "application/json"},
    }
    print("Sending to Gemini API…")
    r = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
    if r.status_code != 200:
        print(f"Gemini error {r.status_code}: {r.text[:300]}")
        return None
    try:
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except Exception as e:
        print("Failed to parse Gemini response:", e)
        return None

def call_mistral(prompt):
    url  = "https://api.mistral.ai/v1/chat/completions"
    data = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }
    print("Sending to Mistral API…")
    r = requests.post(url,
                      headers={"Content-Type": "application/json",
                               "Authorization": f"Bearer {MISTRAL_API_KEY}"},
                      json=data)
    if r.status_code != 200:
        print(f"Mistral error {r.status_code}: {r.text[:300]}")
        return None
    try:
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except Exception as e:
        print("Failed to parse Mistral response:", e)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Scale / swara tables
# ─────────────────────────────────────────────────────────────────────────────
# Values match the Swar-Laya Studio scale selector (actual Sa Hz = baseSa).
# The old file used half-values (×0.5); the HTML migration handles old JSONs.
SCALE_MAP = {
    "Pandhri 1": 261.63, "Kali 1": 277.18, "Pandhri 2": 293.67, "Kali 2": 311.13,
    "Pandhri 3": 329.63, "Pandhri 4": 349.23, "Kali 3": 370.00, "Pandhri 5": 392.00,
    "Kali 4": 415.30,    "Pandhri 6": 440.00, "Kali 5": 466.16, "Pandhri 7": 493.88,
    # Octave-4 note names
    "C4": 261.63, "C#4": 277.18, "D4": 293.67,  "D#4": 311.13, "E4": 329.63,
    "F4": 349.23, "F#4": 370.00, "G4": 392.00,  "G#4": 415.30, "A4": 440.00,
    "A#4": 466.16,"B4": 493.88,  "C5": 523.25,
    # Octave-3 names accepted as-is (user responsible for octave choice)
    "C3": 130.81, "C#3": 138.59, "D3": 146.83,  "D#3": 155.56, "E3": 164.81,
    "F3": 174.61, "F#3": 185.00, "G3": 196.00,  "G#3": 207.65, "A3": 220.00,
    "A#3": 233.08,"B3": 246.94,
}

# 12 swaras, index 0–11 from Sa
SWARA_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "ma", "Pa", "dha", "Dha", "ni", "Ni"]

# Just Intonation frequency ratios (relative to Sa = 1).
# Indian classical music is tuned in JI, not 12-TET.  Mapping detected Hz to
# swaras must use these pure fractional ratios, not equal-temperament positions.
#   12-TET Ga = 400 ¢   JI Ga (5/4) = 386 ¢   → 14 ¢ off
#   12-TET Dha = 900 ¢  JI Dha (5/3) = 884 ¢  → 16 ¢ off
#   12-TET Ni = 1100 ¢  JI Ni (15/8) = 1088 ¢ → 12 ¢ off
JI_RATIOS = [1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8]
_LOG2_JI  = [math.log2(r) for r in JI_RATIOS]   # pre-computed for inner-loop speed

# ─────────────────────────────────────────────────────────────────────────────
# Swara helpers
# ─────────────────────────────────────────────────────────────────────────────
def hz_to_ji_flat(freq_hz, sa_hz):
    """Map freq_hz → flat JI index  (octave * 12 + pitch_class, 0–11).

    Normalises the frequency ratio into the [1, 2) octave, then picks the
    nearest JI ratio using log-space distance.  Returns an integer with the
    same shape as the old round(12*log2(...)) value so the rest of the
    pipeline (octave = idx//12, pitch class = idx%12) is unchanged.
    """
    ratio = freq_hz / sa_hz
    octave = 0
    while ratio >= 2.0:
        ratio /= 2.0
        octave += 1
    while ratio < 1.0:
        ratio *= 2.0
        octave -= 1
    log_r = math.log2(ratio)
    pc = min(range(12), key=lambda i: abs(log_r - _LOG2_JI[i]))
    return octave * 12 + pc

def hz_to_swara(freq_hz, sa_hz):
    """Convert Hz → (swara_name, octave).  Returns ('-', 0) for silence."""
    if not freq_hz or freq_hz <= 0 or sa_hz <= 0 or freq_hz < sa_hz * 0.25:
        return "-", 0
    try:
        idx = hz_to_ji_flat(freq_hz, sa_hz)
        return SWARA_NAMES[idx % 12], idx // 12
    except Exception:
        return "-", 0

def detect_sa(f0, voiced_flag):
    """Estimate the Sa (tonic) frequency from the pyin pitch track.

    Builds a fine log-frequency histogram (3-cent bins) of all voiced frames,
    finds prominent peaks with scipy find_peaks, then scores every SCALE_MAP
    candidate by how much spectral weight falls within 55 cents of it at any
    octave.  Returns (scale_name, hz) for the best match, or (None, None) if
    there are too few voiced frames to make a confident decision.
    """
    voiced_hz = f0[voiced_flag & ~np.isnan(f0) & (f0 > 0)]
    voiced_hz = voiced_hz[(voiced_hz >= 100) & (voiced_hz <= 700)]
    if len(voiced_hz) < 100:
        return None, None

    REF_HZ  = 110.0          # A2 — arbitrary reference for cent calculations
    cents   = (1200.0 * np.log2(voiced_hz / REF_HZ)).astype(int)
    lo, hi  = int(cents.min()) - 10, int(cents.max()) + 10
    edges   = np.arange(lo, hi + 4, 3)   # 3-cent bins
    hist, _ = np.histogram(cents, bins=edges)
    hist_sm = scipy.ndimage.gaussian_filter1d(hist.astype(float), sigma=10)

    peaks, _ = find_peaks(hist_sm, height=hist_sm.max() * 0.07, distance=25)
    if not len(peaks):
        return None, None

    # Centre-of-bin Hz for each peak
    peak_hz = REF_HZ * (2.0 ** ((edges[peaks] + 1.5) / 1200.0))
    peak_h  = hist_sm[peaks]

    # Score each SCALE_MAP entry: sum weighted hits within ±55 cents at any octave
    scale_votes = {}
    for ph, weight in sorted(zip(peak_hz, peak_h), key=lambda x: -x[1])[:20]:
        for name, sa_hz in SCALE_MAP.items():
            if not (180 <= sa_hz <= 530):   # skip octave-3 duplicates
                continue
            for oct_shift in range(-2, 3):
                candidate = sa_hz * (2 ** oct_shift)
                try:
                    diff_cents = abs(1200.0 * math.log2(ph / candidate))
                except Exception:
                    continue
                if diff_cents < 55:
                    scale_votes[name] = (scale_votes.get(name, 0)
                                         + weight * (1.0 - diff_cents / 55.0))

    if not scale_votes:
        return None, None

    top3 = sorted(scale_votes.items(), key=lambda x: -x[1])[:3]
    print("  Sa candidates: "
          + ", ".join(f"{n}={SCALE_MAP[n]:.0f}Hz" for n, _ in top3))
    best_name = top3[0][0]
    return best_name, SCALE_MAP[best_name]


# ─────────────────────────────────────────────────────────────────────────────
# DSP helpers
# ─────────────────────────────────────────────────────────────────────────────
def bandpass_filter(data, low_hz, high_hz, fs):
    nyq  = 0.5 * fs
    b, a = butter(4, [max(0.001, low_hz / nyq), min(0.999, high_hz / nyq)], btype="band")
    return lfilter(b, a, data)

def detect_adlib_splits(y_adlib, sr, min_gap_sec=0.8, threshold_ratio=0.05):
    """Return timestamps (seconds) of silence gaps within the adlib section.

    Slides a 100 ms RMS window; wherever energy stays below threshold_ratio of
    the peak for at least min_gap_sec, that midpoint is returned as a split.
    These are used to suggest where one instrument ends and another begins.
    """
    frame_len   = int(0.1 * sr)
    hop         = frame_len // 2
    rms         = librosa.feature.rms(y=y_adlib, frame_length=frame_len,
                                       hop_length=hop)[0]
    if rms.max() < 1e-8:
        return []

    threshold      = rms.max() * threshold_ratio
    min_gap_frames = max(1, int(min_gap_sec / (hop / sr)))

    splits, in_gap, gap_start = [], False, 0
    for i, r in enumerate(rms):
        if r < threshold and not in_gap:
            in_gap, gap_start = True, i
        elif r >= threshold and in_gap:
            in_gap = False
            if i - gap_start >= min_gap_frames:
                splits.append((gap_start + (i - gap_start) // 2) * hop / sr)

    return splits

def detect_sam_time(y_perc, sr, bpm, beats_per_line):
    """Estimate Sam (taal cycle beat-1) timestamp via onset-vs-click cross-correlation.

    Cross-correlates the percussive onset envelope against a synthetic click
    train at the given BPM to find which shift within the first taal cycle
    gives the best match — that shift is the Sam offset.

    Returns estimated Sam in seconds, or None if signal has no clear periodicity.
    """
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=HOP)
    if onset_env.max() < 1e-6:
        return None

    frames_per_beat = max(1, int(round((60.0 / bpm) * sr / HOP)))
    n               = len(onset_env)

    click = np.zeros(n)
    for i in range(0, n, frames_per_beat):
        click[i] = 1.0

    corr        = np.correlate(onset_env, click, mode='full')
    zero_lag    = n - 1
    line_frames = frames_per_beat * beats_per_line
    window      = corr[zero_lag : zero_lag + line_frames]
    if not len(window):
        return None

    best = int(np.argmax(window))
    if window[best] < window.mean() + window.std():
        return None   # no clear periodicity

    return best * HOP / sr

def correct_octave_errors(f0, sa_hz):
    """Double pyin frames that landed an octave below the true pitch.
    Runs twice to handle double-octave errors at very low values."""
    out = f0.copy()
    for _ in range(2):
        mask = ~np.isnan(out) & (out > 0) & (out < sa_hz / 1.8)
        out[mask] *= 2
    return out

def smooth_pitch(f0, w=5):
    """Median-filter the pitch track (voiced frames only) to remove vibrato noise."""
    out = f0.copy()
    for i in range(len(f0)):
        if np.isnan(f0[i]) or f0[i] <= 0:
            continue
        lo = max(0, i - w // 2)
        hi = min(len(f0), i + w // 2 + 1)
        seg = f0[lo:hi]
        voiced_seg = seg[~np.isnan(seg) & (seg > 0)]
        if len(voiced_seg):
            out[i] = np.median(voiced_seg)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Beat-level note aggregation
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_beat(f0_seg, voiced_seg, sa_hz, min_sub_fraction=0.20):
    """Return (notes_list, octs_list) for one beat window of pyin frames.

    Four-step pipeline designed to survive Indian classical vibrato (andolan):

    1. Convert voiced Hz → rounded semitone indices relative to sa_hz.
    2. Dominant note by majority vote — immune to the temporal ordering of
       vibrato up/down strokes that makes plain RLE pick the wrong note.
    3. Vibrato absorption — snap every frame within ±1 semitone of the
       dominant back to the dominant.  A ga (3) that oscillates ±1 to Ga (4)
       during andolan must never be misread as Ga.
    4. RLE on the cleaned array — now only genuine sub-note transitions
       (gamak, meend endpoints) survive; keep runs ≥ min_sub_fraction.
    """
    mask = voiced_seg & ~np.isnan(f0_seg) & (f0_seg > 0)
    voiced_hz = f0_seg[mask]

    if len(voiced_hz) == 0:
        return ["-"], [0]

    semitone_idxs = []
    for h in voiced_hz:
        try:
            semitone_idxs.append(hz_to_ji_flat(h, sa_hz))
        except Exception:
            pass

    if not semitone_idxs:
        return ["-"], [0]

    arr   = np.array(semitone_idxs)
    total = len(arr)
    min_frames = max(1, int(min_sub_fraction * total))

    # ── Step 2: dominant by majority vote ────────────────────────────────────
    dominant = int(Counter(arr).most_common(1)[0][0])

    # ── Step 3: vibrato absorption ────────────────────────────────────────────
    # Frames within ±1 semitone of dominant are the same note with vibrato.
    # Snap them to dominant so they don't generate spurious RLE transitions.
    clean = np.where(np.abs(arr - dominant) <= 1, dominant, arr)

    # ── Step 4: RLE on cleaned array ─────────────────────────────────────────
    rle = []
    curr, cnt = int(clean[0]), 1
    for s in clean[1:]:
        s = int(s)
        if s == curr:
            cnt += 1
        else:
            rle.append((curr, cnt))
            curr, cnt = s, 1
    rle.append((curr, cnt))

    significant = [(s, c) for s, c in rle if c >= min_frames]
    if not significant:
        significant = [(dominant, 1)]

    notes = [SWARA_NAMES[s % 12] for s, _ in significant]
    octs  = [s // 12 for s, _ in significant]
    return notes, octs

def beat_diagnostics(f0_seg, voiced_seg, sa_hz):
    """Return raw pyin statistics for one beat window — used for display only.

    Tells the user what Hz values pyin actually found, before any note-mapping
    or sustain logic, so they can verify whether the issue is in pitch detection
    or in note assignment.

    Returns a dict:
        voiced_pct  : int 0–100  — % of frames pyin marked as voiced
        clusters    : list of up to 3 dicts, descending by frame count, each:
                        hz   (float) — centre Hz of that semitone bucket
                        note (str)   — SWARA_NAMES mapping for that Hz
                        oct  (int)   — octave relative to Sa
                        pct  (int)   — % of voiced frames in this cluster
    """
    total = len(f0_seg)
    if total == 0:
        return {"voiced_pct": 0, "clusters": []}

    mask       = voiced_seg & ~np.isnan(f0_seg) & (f0_seg > 0)
    voiced_hz  = f0_seg[mask]
    voiced_pct = int(100 * len(voiced_hz) / total)

    if len(voiced_hz) == 0:
        return {"voiced_pct": voiced_pct, "clusters": []}

    semitones = []
    for h in voiced_hz:
        try:
            semitones.append(hz_to_ji_flat(h, sa_hz))
        except Exception:
            pass

    if not semitones:
        return {"voiced_pct": voiced_pct, "clusters": []}

    arr     = np.array(semitones)
    counter = Counter(arr)
    total_v = len(arr)

    clusters = []
    for sem, cnt in counter.most_common(3):
        pc   = sem % 12
        oct  = sem // 12
        hz   = sa_hz * JI_RATIOS[pc] * (2.0 ** oct)   # JI centre Hz, not 12-TET
        note = SWARA_NAMES[pc]
        pct  = int(100 * cnt / total_v)
        clusters.append({"hz": hz, "note": note, "oct": oct, "pct": pct})

    return {"voiced_pct": voiced_pct, "clusters": clusters}


# ─────────────────────────────────────────────────────────────────────────────
# Instrument classification helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_instrument_profiles(samples_dir):
    profiles = {}
    if not os.path.exists(samples_dir):
        return profiles
    print(f"Building instrument profiles from {samples_dir}…")
    for folder in os.listdir(samples_dir):
        inst_path = os.path.join(samples_dir, folder)
        if not os.path.isdir(inst_path):
            continue
        files = [f for f in os.listdir(inst_path) if f.endswith(".mp3")][:3]
        mfccs = []
        for fname in files:
            try:
                ys, srs = librosa.load(os.path.join(inst_path, fname), sr=22050, mono=True)
                mfccs.append(np.mean(librosa.feature.mfcc(y=ys, sr=srs, n_mfcc=13), axis=1))
            except Exception:
                pass
        if mfccs:
            profiles[folder.lower()] = np.mean(mfccs, axis=0)
    if profiles:
        print(f"Profiles built for: {', '.join(profiles)}")
    return profiles

def classify_instrument(audio_chunk, sr, profiles, allowed_insts=None):
    if not profiles or len(audio_chunk) < sr * 0.1:
        return "harmonium"
    try:
        if sr != 22050:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=22050)
        prof = np.mean(librosa.feature.mfcc(y=audio_chunk, sr=22050, n_mfcc=13), axis=1)
        best, min_d = "harmonium", float("inf")
        for inst, p in profiles.items():
            if allowed_insts and inst not in allowed_insts:
                continue
            d = np.linalg.norm(prof - p)
            if d < min_d:
                min_d, best = d, inst
        return best
    except Exception:
        return "harmonium"

# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_line(notes, octs, instrument, beats_per_line, adlib=False):
    return {
        "section":         "",
        "line_instrument": instrument,
        "percussion":      "mute" if adlib else "tabla",
        "line_volume":     1.0,
        "adlib":           adlib,
        "legato":          adlib,
        "line_bpm":        None,
        "taal_key":        str(beats_per_line),
        "beats":           len(notes),
        "lyrics":          [[""] for _ in notes],
        "notes":           notes,
        "octaves":         octs,
        "meend":           [[False] * len(b) for b in notes],
    }

def extract_essential_data(lines):
    return [{"adlib": l.get("adlib", False), "beats": l.get("beats", 0),
             "notes": l.get("notes", []),    "octaves": l.get("octaves", [])}
            for l in lines]

# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE HELPERS  (module-level — no dependency on song data)
# ═══════════════════════════════════════════════════════════════════════════════
RATE           = 22050
HOP            = 512
FRAME_LEN      = 2048
STRETCH_FACTOR = 3.0   # prolongation: stretch audio N× before pyin so fast ornaments stabilise

NOTE_TO_SEMITONE = {
    "Sa": 0, "re": 1, "Re": 2, "ga": 3, "Ga": 4, "Ma": 5,
    "ma": 6, "Pa": 7, "dha": 8, "Dha": 9, "ni": 10, "Ni": 11,
}
VALID_NOTES = set(NOTE_TO_SEMITONE) | {"-", "~"}

try:
    import sounddevice as _sd
    def play_audio(arr, sr=RATE):
        """Play a float32 numpy array; blocks until done."""
        _sd.play(arr.astype(np.float32), samplerate=sr)
        _sd.wait()
except ImportError:
    def play_audio(arr, sr=RATE):
        print("  [sounddevice not installed — skipping playback]")
        print("  Install with:  pip install sounddevice")

def synth_line(notes_list, octs_list, sa_hz, beat_dur_sec, sr=RATE):
    """Render a notation segment as audio using a simple harmonium-like tone."""
    out = []
    last_freq = None
    for beat_notes, beat_octs in zip(notes_list, octs_list):
        sub_dur = beat_dur_sec / max(1, len(beat_notes))
        n_samp  = max(1, int(sub_dur * sr))
        for note, oct in zip(beat_notes, beat_octs):
            t = np.linspace(0, sub_dur, n_samp, endpoint=False)
            if note == "-":
                out.append(np.zeros(n_samp, dtype=np.float32))
                last_freq = None
            elif note == "~":
                freq = last_freq or 0.0
                if freq > 0:
                    wave = (0.25 * np.sin(2*np.pi*freq*t)
                          + 0.12 * np.sin(2*np.pi*2*freq*t)
                          + 0.06 * np.sin(2*np.pi*3*freq*t))
                    out.append(wave.astype(np.float32))
                else:
                    out.append(np.zeros(n_samp, dtype=np.float32))
            else:
                sem  = NOTE_TO_SEMITONE.get(note, 0)
                freq = sa_hz * (2.0 ** (oct + sem / 12.0))
                wave = (0.25 * np.sin(2*np.pi*freq*t)
                      + 0.12 * np.sin(2*np.pi*2*freq*t)
                      + 0.06 * np.sin(2*np.pi*3*freq*t))
                att = min(int(0.015 * sr), n_samp // 4)
                if att > 0:
                    wave[:att] *= np.linspace(0.0, 1.0, att)
                out.append(wave.astype(np.float32))
                last_freq = freq
    return np.concatenate(out) if out else np.zeros(int(beat_dur_sec * sr),
                                                     dtype=np.float32)

def display_notation(notes_list, octs_list, cols=8, diags=None):
    """Print a formatted beat grid.

    When `diags` (list of beat_diagnostics dicts) is supplied, two extra rows
    are shown per row-group so the user can see exactly what pyin found:

        Note   — the transcribed note that will go into the JSON
        pyin   — dominant Hz detected + the note it maps to (raw, before sustain)
        voiced — % of frames pyin marked as voiced (low = noisy / silent beat)
                 if a second cluster has ≥ 25 % weight it is shown as alt/note%
    """
    OCT_SYM = {-2: "‥", -1: ".", 0: "", 1: "'", 2: "''"}
    n    = len(notes_list)
    CW   = 9   # column width (fits "494/dha" = 7 chars with padding)
    LW   = 10  # label width

    for row_s in range(0, n, cols):
        row_e = min(row_s + cols, n)
        idx   = range(row_s, row_e)

        # ── Beat numbers ──────────────────────────────────────────────────────
        print(" " * LW + "".join(f"{i+1:>{CW}}" for i in idx))

        # ── Transcribed notes ─────────────────────────────────────────────────
        note_cells = []
        for bn, bo in zip(notes_list[row_s:row_e], octs_list[row_s:row_e]):
            if len(bn) == 1:
                pfx = OCT_SYM.get(bo[0], str(bo[0]))
                note_cells.append(pfx + bn[0])
            else:
                parts = [OCT_SYM.get(o, str(o)) + nn for nn, o in zip(bn, bo)]
                note_cells.append("/".join(parts))
        print(f"{'Note:':>{LW}}" + "".join(f"{s:>{CW}}" for s in note_cells))

        # ── Diagnostic rows ───────────────────────────────────────────────────
        if diags:
            row_d = diags[row_s:row_e]

            # pyin: dominant Hz → note mapping
            pyin_cells = []
            for d in row_d:
                if d["clusters"]:
                    c   = d["clusters"][0]
                    pfx = OCT_SYM.get(c["oct"], str(c["oct"]))
                    pyin_cells.append(f"{c['hz']:.0f}/{pfx}{c['note']}")
                else:
                    pyin_cells.append("---")
            print(f"{'pyin:':>{LW}}" + "".join(f"{s:>{CW}}" for s in pyin_cells))

            # voiced%: signal strength + second cluster if significant
            info_cells = []
            for d in row_d:
                cell = f"{d['voiced_pct']}%"
                if len(d["clusters"]) > 1 and d["clusters"][1]["pct"] >= 25:
                    c2  = d["clusters"][1]
                    pfx = OCT_SYM.get(c2["oct"], str(c2["oct"]))
                    cell += f" {pfx}{c2['note']}{c2['pct']}%"
                info_cells.append(cell)
            print(f"{'voiced:':>{LW}}" + "".join(f"{s:>{CW}}" for s in info_cells))

        print()

def edit_notation(notes_list, octs_list):
    """Interactively patch beats.  Returns (notes_list, octs_list).

    Syntax
    ------
    <start_beat>  <note> <note> <note> …

    One command sets consecutive beats from start_beat onwards.
    Octave is indicated by a PREFIX on each note (Bhatkhande convention):

        note       madhya saptak   (octave  0)  — no prefix
        .note      mandra saptak   (octave -1)  — one dot
        ..note     ati-mandra      (octave -2)  — two dots
        'note      taar saptak     (octave +1)  — apostrophe
        ''note     ati-taar        (octave +2)  — two apostrophes

    Sub-beat gamak: separate two notes with /   e.g.  ga/Ma  or  .ga/'Ma
    Silence: -       Sustain: ~

    Examples:
      1 .dha .ni Sa ga Ma         beats 1-5: mandra dha ni, then madhya Sa ga Ma
      3 'Sa                       beat 3 = taar Sa
      6 ga/Ma                     beat 6 = gamak ga→Ma (both madhya)
      6 .ga/'Ma                   beat 6 = gamak mandra-ga to taar-Ma
      4 - - Sa                    beats 4-6: silence silence Sa
      done                        finish editing
    """

    def _oct_pfx(o):
        return {-2: "..", -1: ".", 0: "", 1: "'", 2: "''"}.get(o, str(o))

    def _parse_tok(tok):
        """Parse one token like '.dha', \"'Sa\", 'ga/Ma', '.ga/\\'Ma' →
        ([notes], [octs]) or raise ValueError."""
        result_n, result_o = [], []
        for part in tok.split("/"):
            if part.startswith("''"):
                ov, part = 2, part[2:]
            elif part.startswith("'"):
                ov, part = 1, part[1:]
            elif part.startswith(".."):
                ov, part = -2, part[2:]
            elif part.startswith("."):
                ov, part = -1, part[1:]
            else:
                ov = 0
            if part not in VALID_NOTES:
                raise ValueError(part)
            result_n.append(part)
            result_o.append(ov)
        return result_n, result_o

    print("\n  Edit mode  (type 'done' to finish)\n")
    print("  Octave prefix: .note=mandra  note=madhya  'note=taar  ''note=ati-taar")
    print("  Sequence:  <beat> <note> <note> …   sets consecutive beats")
    print("  Gamak:     ga/Ma  or  .ga/'Ma")
    print("  Example:   1 .dha .ni Sa ga Ma   →  beats 1-5 across octaves\n")

    while True:
        cmd = input("  Edit> ").strip()
        if not cmd or cmd.lower() == "done":
            break

        parts = cmd.split()
        if len(parts) < 2:
            print("    ? Need at least: <beat> <note>"); continue

        try:
            start_bi = int(parts[0]) - 1
        except ValueError:
            print("    ? First token must be a beat number"); continue
        if start_bi < 0 or start_bi >= len(notes_list):
            print(f"    ? Beat must be 1–{len(notes_list)}"); continue

        note_tokens = parts[1:]

        # ── Backward-compat: "3 Ma -1"  (single note + explicit integer octave)
        if (len(note_tokens) == 2
                and "/" not in note_tokens[0]
                and not note_tokens[0][0] in (".", "'")
                and note_tokens[0] in VALID_NOTES
                and note_tokens[1].lstrip("-").isdigit()):
            try:
                ov = int(note_tokens[1])
            except ValueError:
                print("    ? Octave must be an integer"); continue
            notes_list[start_bi] = [note_tokens[0]]
            octs_list[start_bi]  = [ov]
            print(f"    Beat {start_bi+1} → {_oct_pfx(ov)}{note_tokens[0]}")
            continue

        # ── Sequence mode: one token per beat from start_bi onwards
        parsed = []
        ok = True
        for tok in note_tokens:
            try:
                nn, oo = _parse_tok(tok)
                parsed.append((nn, oo))
            except ValueError as e:
                print(f"    ? Unknown note '{e}'.  Valid: "
                      f"{sorted(n for n in VALID_NOTES if n not in ('-','~'))}"
                      f" + '-' (silence) '~' (sustain)")
                ok = False
                break
        if not ok:
            continue

        for j, (nn, oo) in enumerate(parsed):
            bi = start_bi + j
            if bi >= len(notes_list):
                print(f"    (reached end of segment at beat {len(notes_list)})")
                break
            notes_list[bi] = nn
            octs_list[bi]  = oo
            note_str = "/".join(_oct_pfx(o) + n for n, o in zip(nn, oo))
            print(f"    Beat {bi+1} → {note_str}")

    return notes_list, octs_list

def last_sounding_note(notes_list, octs_list):
    """Return the last non-silence, non-sustain (note, oct) lists, or (None, None)."""
    for i in range(len(notes_list) - 1, -1, -1):
        if notes_list[i] not in (["-"], ["~"]):
            return list(notes_list[i]), list(octs_list[i])
    return None, None

# ═══════════════════════════════════════════════════════════════════════════════
#  INITIALISATION — load audio, run pyin, confirm Sa
# ═══════════════════════════════════════════════════════════════════════════════
try:
    audio_file   = input("Audio filename (e.g. song.mp3): ").strip()
    tempo_str    = input("Tempo factor [1.0]: ").strip()
    tempo_factor = float(tempo_str) if tempo_str else 1.0

    profiles = build_instrument_profiles("public/samples")

    print("\nLoading audio…")
    y, sr = librosa.load(audio_file, sr=RATE, mono=True)

    dur_str = input("Max seconds to process (Enter = full song): ").strip()
    if dur_str:
        y = y[:int(float(dur_str) * sr)]
        print(f"Trimmed to {float(dur_str):.0f} s.")

    if tempo_factor != 1.0:
        print(f"Time-stretching ×{tempo_factor}…")
        y = librosa.effects.time_stretch(y, rate=tempo_factor)

    total_dur_sec = len(y) / RATE

    print("Separating harmonic/percussive layers (HPSS)…")
    y_harm, y_perc = librosa.effects.hpss(y, margin=3.0)

    song_name_in = input("Song name: ").strip()

    song_bpm      = int(input("BPM: "))
    effective_bpm = song_bpm * tempo_factor
    beat_dur_sec  = 60.0 / effective_bpm

    beats_str      = input("Beats per taal line — Trital=16, Keherwa=8, Rupak=7 [16]: ").strip()
    beats_per_line = int(beats_str) if beats_str else 16

    smooth_w     = 9 if song_bpm < 100 else 5
    min_sub_frac = 0.30 if song_bpm < 100 else 0.20

    print("Detecting taal start (Sam) from percussive onsets…")
    auto_sam = detect_sam_time(y_perc, RATE, song_bpm, beats_per_line)
    hint_sam = (f" [auto-detected: {auto_sam:.2f} s]"
                if auto_sam is not None else " [not detected — enter manually]")
    sam_str  = input(f"Sam / taal start time (s){hint_sam} [Enter = 0]: ").strip()
    sam_time = float(sam_str) if sam_str else (auto_sam if auto_sam is not None else 0.0)
    print(f"Sam = {sam_time:.2f} s")

    # Wide-range pyin first, then Sa detection, then narrow correction
    y_filt = bandpass_filter(y_harm, 100.0, 700.0, RATE)

    # Prolongation: stretch the filtered audio STRETCH_FACTOR× so PYIN sees each
    # note held for more frames — fast taans and ornaments stabilise into clear pitches.
    # The stretched output is then resampled back to the original frame grid so the
    # rest of the pipeline (beat timing, playback sync) is unchanged.
    print(f"Time-stretching {STRETCH_FACTOR:.0f}× for better pitch detection…")
    y_filt_stretched = librosa.effects.time_stretch(y_filt, rate=1.0 / STRETCH_FACTOR)
    n_orig = max(1, 1 + (len(y_filt) - FRAME_LEN) // HOP)
    print(f"Running pyin on stretched audio (~{round(STRETCH_FACTOR * 60)} s for a 5-min song)…")
    f0_stretched, voiced_stretched, _ = librosa.pyin(
        y_filt_stretched, fmin=100.0, fmax=700.0, sr=RATE,
        frame_length=FRAME_LEN, hop_length=HOP, fill_na=np.nan,
    )

    # Resample stretched frame array back to original frame count
    n_stretch   = len(f0_stretched)
    scale       = n_stretch / max(1, n_orig)
    f0_raw      = np.full(n_orig, np.nan)
    voiced_flag = np.zeros(n_orig, dtype=bool)
    for i in range(n_orig):
        lo = int(i * scale)
        hi = min(int((i + 1) * scale), n_stretch)
        if lo >= n_stretch:
            break
        seg_f      = f0_stretched[lo:hi]
        seg_v      = voiced_stretched[lo:hi]
        voiced_sub = seg_f[seg_v & ~np.isnan(seg_f) & (seg_f > 0)]
        if len(voiced_sub) >= max(1, int(len(seg_f) * 0.3)):
            f0_raw[i]     = np.median(voiced_sub)
            voiced_flag[i] = True
    print("Detecting Sa from pitch distribution…")
    auto_sa_name, auto_sa_hz = detect_sa(f0_raw, voiced_flag)

    hint_str = (f" [{auto_sa_name} = {auto_sa_hz:.0f} Hz auto-detected]"
                if auto_sa_name else "")
    scale_in = input(
        f"Sa scale{hint_str}\n"
        "  (Enter = accept auto, or type 'Kali 2' / 'E4' / Hz to override): "
    ).strip()
    if not scale_in:
        SA_HZ = auto_sa_hz if auto_sa_hz else 311.13
        print(f"Sa = {SA_HZ:.2f} Hz  (auto: {auto_sa_name or 'fallback'})")
    else:
        try:
            SA_HZ = float(scale_in)
        except ValueError:
            SA_HZ = SCALE_MAP.get(scale_in, auto_sa_hz or 311.13)
        print(f"Sa = {SA_HZ:.2f} Hz")

    f0_corr   = correct_octave_errors(f0_raw, SA_HZ)
    f0_smooth = smooth_pitch(f0_corr, w=smooth_w)
    SEC_PER_FRAME = HOP / RATE

    valid_f0 = f0_smooth[voiced_flag & ~np.isnan(f0_smooth)]
    if len(valid_f0):
        print(f"pyin: {voiced_flag.sum()}/{len(voiced_flag)} voiced frames | "
              f"F0 {valid_f0.min():.0f}–{valid_f0.max():.0f} Hz | "
              f"median {np.median(valid_f0):.0f} Hz")

    # ── Segment helper (closes over f0_smooth, voiced_flag, SA_HZ, …) ─────────
    def process_segment(start_sec, n_beats):
        """Extract notes, octaves, and per-beat diagnostics for n_beats.

        Returns (notes_list, octs_list, diags_list).
        diags_list is a list of beat_diagnostics dicts — raw pyin statistics
        that are shown alongside the transcribed notes so the user can see
        exactly what frequencies were detected at each beat.
        """
        nl, ol, dl = [], [], []
        for i in range(n_beats):
            bs   = start_sec + i * beat_dur_sec
            be   = bs + beat_dur_sec
            fi_s = max(0, int(bs / SEC_PER_FRAME))
            fi_e = min(len(f0_smooth), int(be / SEC_PER_FRAME))
            if fi_e <= fi_s:
                nl.append(["-"]); ol.append([0])
                dl.append({"voiced_pct": 0, "clusters": []})
            else:
                f0_win = f0_smooth[fi_s:fi_e]
                v_win  = voiced_flag[fi_s:fi_e]
                nn, oo = aggregate_beat(f0_win, v_win, SA_HZ,
                                        min_sub_fraction=min_sub_frac)
                nl.append(nn); ol.append(oo)
                dl.append(beat_diagnostics(f0_win, v_win, SA_HZ))
        return nl, ol, dl

    def apply_sustain_to_segment(nl, ol, carry_n, carry_o):
        """Replace repeated beats with ~ sustain.  Returns (nl, ol, last_n, last_o)."""
        rn, ro = [], []
        last_n, last_o = carry_n, carry_o
        for bn, bo in zip(nl, ol):
            if bn == ["-"]:
                rn.append(bn); ro.append(bo)
                last_n = last_o = None
            elif last_n is not None and bn == last_n and bo == last_o:
                rn.append(["~"]); ro.append([0])
            else:
                rn.append(list(bn)); ro.append(list(bo))
                last_n, last_o = list(bn), list(bo)
        return rn, ro, last_n, last_o

except Exception as e:
    print(f"Init error: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE TRANSCRIPTION LOOP
#  Mimics the guru-shishya method: one phrase at a time, play → verify → accept
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"  {song_name_in or 'Song'}  |  Sa = {SA_HZ:.1f} Hz  |  "
      f"BPM = {song_bpm}  |  {beats_per_line} beats/line")
print(f"  Song duration: {total_dur_sec:.1f} s  |  Beat = {beat_dur_sec:.3f} s")
print(f"  Commands: (a)ccept  (e)dit  (r)e-segment  (s)kip  (q)uit+save")
print(f"{'='*62}\n")

all_lines    = []
position_sec = 0.0
carry_n      = None   # last accepted note — carried into next segment for ~
carry_o      = None

# ── Adlib section before Sam ──────────────────────────────────────────────────
# Step through the pre-taal portion as multiple sub-segments (one per instrument).
# Silence gaps are auto-detected to suggest where to split; the user can adjust.
if sam_time > beat_dur_sec * 0.5:
    y_adlib     = y[: int(sam_time * RATE)]
    adlib_splits = detect_adlib_splits(y_adlib, RATE)

    print(f"\n{'='*62}")
    print(f"  Adlib  0 → {sam_time:.2f} s  |  Commands: (a)ccept (e)dit (r)e-segment (s)kip (q)done")
    if adlib_splits:
        print(f"  Suggested instrument change(s) at: "
              + ", ".join(f"{t:.2f}s" for t in adlib_splits))
    print(f"{'='*62}")

    adlib_pos = 0.0

    while adlib_pos < sam_time - beat_dur_sec * 0.5:
        remaining_adlib = sam_time - adlib_pos

        # Suggest the next natural boundary (next silence gap or sam_time)
        next_split  = next((t for t in adlib_splits if t > adlib_pos + 0.1), sam_time)
        default_end = min(next_split, sam_time)
        n_default   = max(1, round((default_end - adlib_pos) / beat_dur_sec))

        print(f"\n── Adlib pos {adlib_pos:.1f}s  ({remaining_adlib:.1f}s to Sam) ──")

        start_str = input(f"  Start (s) [{adlib_pos:.1f}]: ").strip()
        try:
            seg_start = float(start_str) if start_str else adlib_pos
        except ValueError:
            seg_start = adlib_pos

        end_hint = f"{default_end:.1f}s = {n_default} beats"
        seg_str  = input(f"  End (s) or n-beats  [{end_hint}]: ").strip()
        if not seg_str:
            n_beats, seg_end = n_default, default_end
        else:
            try:
                val = float(seg_str)
                if val <= 64 and seg_str.isdigit():
                    n_beats = int(val)
                    seg_end = seg_start + n_beats * beat_dur_sec
                else:
                    seg_end = min(val, sam_time)
                    n_beats = max(1, round((seg_end - seg_start) / beat_dur_sec))
            except ValueError:
                n_beats, seg_end = n_default, default_end
        seg_end = min(seg_end, sam_time)
        n_beats = max(1, round((seg_end - seg_start) / beat_dur_sec))

        # Instrument suggestion via MFCC if profiles are available
        inst_hint = ""
        if profiles:
            chunk     = y[int(seg_start * RATE): int(seg_end * RATE)]
            suggested = classify_instrument(chunk, RATE, profiles)
            inst_hint = f" [suggested: {suggested}]"
        inst_in    = input(f"  Instrument{inst_hint} [harmonium]: ").strip() or "harmonium"
        section_in = input("  Section label [Adlib]: ").strip() or "Adlib"

        notes_ad, octs_ad, diags_ad = process_segment(seg_start, n_beats)

        while True:
            notes_sus_ad, octs_sus_ad, _, _ = apply_sustain_to_segment(
                notes_ad, octs_ad, carry_n, carry_o)

            print(f"\n  ┌─ Adlib  {n_beats} beat(s)  {seg_start:.1f}s → {seg_end:.1f}s  [{inst_in}] ─┐")
            display_notation(notes_sus_ad, octs_sus_ad, cols=8, diags=diags_ad)

            ans = input("  ▶  Play original? [Enter=yes / n]: ").strip().lower()
            if ans != "n":
                play_audio(y[int(seg_start * RATE): int(seg_end * RATE)], RATE)

            ans = input("  ♪  Play synthesized? [Enter=yes / n]: ").strip().lower()
            if ans != "n":
                play_audio(synth_line(notes_sus_ad, octs_sus_ad, SA_HZ, beat_dur_sec), RATE)

            dec = input(
                "\n  → (a)ccept  (e)dit  (r)e-segment  (s)kip  (q)done-with-adlib : "
            ).strip().lower()

            if dec == "a":
                all_lines.append({
                    "section":         section_in,
                    "line_instrument": inst_in,
                    "percussion":      "mute",
                    "line_volume":     1.0,
                    "adlib":           True,
                    "legato":          True,
                    "line_bpm":        None,
                    "taal_key":        "",
                    "beats":           n_beats,
                    "lyrics":          [[""] for _ in notes_sus_ad],
                    "notes":           notes_sus_ad,
                    "octaves":         octs_sus_ad,
                    "meend":           [[False] * len(b) for b in notes_sus_ad],
                })
                carry_n, carry_o = last_sounding_note(notes_sus_ad, octs_sus_ad)
                adlib_pos = seg_end
                print(f"  ✓ Accepted [{inst_in}] — {len(all_lines)} line(s) total.")
                break

            elif dec == "e":
                notes_sus_ad, octs_sus_ad = edit_notation(notes_sus_ad, octs_sus_ad)

            elif dec == "r":
                break   # re-enter segment boundaries for this sub-segment

            elif dec == "s":
                adlib_pos = seg_end
                break

            elif dec == "q":
                adlib_pos = sam_time   # exit adlib loop entirely
                break

    position_sec = sam_time
    print(f"\n  Taal starts at {sam_time:.2f} s — entering rhythmic section.")

while True:
    remaining = total_dur_sec - position_sec
    if remaining < beat_dur_sec * 0.5:
        print("End of song reached.")
        break

    print(f"\n── Position {position_sec:.1f} s  ({remaining:.1f} s remaining) "
          f"─ {len(all_lines)} line(s) accepted so far ──")

    # ── Segment boundaries ────────────────────────────────────────────────────
    start_str = input(f"  Start (s) [{position_sec:.1f}]: ").strip()
    try:
        start_sec = float(start_str) if start_str else position_sec
    except ValueError:
        start_sec = position_sec

    default_end = start_sec + beats_per_line * beat_dur_sec
    seg_str = input(
        f"  Beats or end-time (s)  [{beats_per_line} beats = {default_end:.1f} s]: "
    ).strip()
    if not seg_str:
        n_beats = beats_per_line
    else:
        try:
            val = float(seg_str)
            # Heuristic: integers ≤ 64 are beat counts, larger floats are seconds
            if val <= 64 and seg_str.isdigit():
                n_beats = int(val)
            else:
                n_beats = max(1, round((val - start_sec) / beat_dur_sec))
        except ValueError:
            n_beats = beats_per_line
    max_possible = max(1, int((total_dur_sec - start_sec) / beat_dur_sec) + 1)
    n_beats  = max(1, min(n_beats, max_possible))
    end_sec  = start_sec + n_beats * beat_dur_sec

    adlib_ans  = input("  Adlib / prelude? (y/n) [n]: ").strip().lower()
    is_adlib   = (adlib_ans == "y")
    inst_ans   = input("  Instrument [harmonium]: ").strip() or "harmonium"
    section_ans = input("  Section label (e.g. Mukhda, Antara) []: ").strip()

    # ── Process → display → play → decide (inner loop) ───────────────────────
    seg_carry_n = carry_n   # snapshot: reset to this if user re-segments
    seg_carry_o = carry_o

    notes_raw, octs_raw, diags = process_segment(start_sec, n_beats)
    notes_sus, octs_sus, _, _ = apply_sustain_to_segment(
        notes_raw, octs_raw, seg_carry_n, seg_carry_o)

    while True:
        print(f"\n  ┌─ {n_beats} beat(s)  {start_sec:.1f} s → {end_sec:.1f} s ─┐")
        display_notation(notes_sus, octs_sus, cols=8, diags=diags)

        # Play original
        ans = input("  ▶  Play original audio? [Enter=yes / n]: ").strip().lower()
        if ans != "n":
            seg_audio = y[int(start_sec * RATE) : int(end_sec * RATE)]
            play_audio(seg_audio, RATE)

        # Play synthesis
        ans = input("  ♪  Play synthesized notation? [Enter=yes / n]: ").strip().lower()
        if ans != "n":
            synth = synth_line(notes_sus, octs_sus, SA_HZ, beat_dur_sec)
            play_audio(synth, RATE)

        decision = input(
            "\n  → (a)ccept  (e)dit  (r)e-segment  (s)kip  (q)uit+save : "
        ).strip().lower()

        if decision == "a":
            line = {
                "section":         section_ans,
                "line_instrument": inst_ans,
                "percussion":      "mute" if is_adlib else "tabla",
                "line_volume":     1.0,
                "adlib":           is_adlib,
                "legato":          is_adlib,
                "line_bpm":        None,
                "taal_key":        str(beats_per_line),
                "beats":           n_beats,
                "lyrics":          [[""] for _ in notes_sus],
                "notes":           notes_sus,
                "octaves":         octs_sus,
                "meend":           [[False] * len(b) for b in notes_sus],
            }
            all_lines.append(line)
            carry_n, carry_o = last_sounding_note(notes_sus, octs_sus)
            position_sec = end_sec
            print(f"  ✓ Accepted — {len(all_lines)} line(s) total.")
            break

        elif decision == "e":
            notes_sus, octs_sus = edit_notation(notes_sus, octs_sus)
            # Loop back: re-display, re-play, re-decide

        elif decision == "r":
            print("  Re-entering segment boundaries…")
            break   # exit inner loop → outer loop re-prompts start/end

        elif decision == "s":
            position_sec = end_sec
            print(f"  Skipped to {position_sec:.1f} s.")
            break

        elif decision == "q":
            print("  Stopping.")
            position_sec = total_dur_sec   # trigger outer loop exit
            break

    if position_sec >= total_dur_sec:
        break   # 'q' or natural end

# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE + OPTIONAL AI CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════
if not all_lines:
    print("No lines accepted — nothing to save.")
    sys.exit(0)

final_output = {
    "title": song_name_in or "Transcription",
    "scale": str(SA_HZ),
    "bpm":   song_bpm,
    "lines": all_lines,
}
with open("precision_output.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2)
print(f"\n{len(all_lines)} line(s) saved → precision_output.json")

use_ai = input("\nRun AI cleanup on accepted lines? (y/n): ").strip().lower()
if use_ai != "y":
    sys.exit(0)

raw_lines = extract_essential_data(final_output["lines"])
slow_note = ("This is a SLOW song — expect many consecutive identical notes "
             "that should be collapsed to ~ sustains."
             if song_bpm < 110 else "")

prompt = f"""
You are an expert Indian Classical musician and transcriber.
I have automatically extracted raw vocal pitch data from an audio recording.
Errors in the raw data include: vibrato mis-detected as sub-notes, octave jumps,
and adjacent-semitone noise around true pitches.

SONG CONTEXT
  Title  : {song_name_in or "unknown"}
  BPM    : {song_bpm}  {slow_note}
  Sa     : {SA_HZ:.2f} Hz

SWARA SYSTEM (12 semitones from Sa, octave 0 = madhya):
  Index  : 0   1   2   3   4   5   6    7   8    9    10  11
  Swara  : Sa  re  Re  ga  Ga  Ma  ma   Pa  dha  Dha  ni  Ni
  Positive octave = taar saptak; negative = mandra saptak.

RULES — apply ALL without exception:

1. VIBRATO/ANDOLAN REMOVAL
   A beat containing sub-notes that are adjacent semitones (e.g. ["Ga","Ma","Ga"]
   or ["dha","Pa","dha"]) is ONE held note with vibrato.
   Action: collapse to the DOMINANT semitone by majority vote → single-element array.

2. SUSTAIN SUBSTITUTION
   If the same note appears in consecutive beats, keep it as the note on the
   first beat and replace every subsequent beat (until a new note) with "~".
   The octave of every "~" beat MUST be [0].
   {f"For a {song_bpm} BPM slow song, expect long runs of ~ — do not be afraid to use them." if song_bpm < 110 else ""}

3. SUB-BEAT CONSOLIDATION
   Arrays like ["Sa","~","~"] or ["Sa","-","-"] → ["Sa"].
   Keep a sub-beat array ONLY when two or more genuinely DIFFERENT notes occur
   within that beat (a quick phrase/gamak/taan), e.g. ["Ga","Ma"] or ["Pa","Ni"].

4. OCTAVE CONSISTENCY
   Adjacent beats should not jump more than one octave unless clearly intentional.
   Isolated mandra (negative octave) notes surrounded by madhya notes are likely
   pyin octave-detection errors — correct them to octave 0.

5. ADLIB SPLITTING
   A long adlib line containing a silence gap (2+ consecutive "-" beats) may be
   split into separate shorter lines at that gap.
   CONSTRAINT: total beats across split lines = original total beats.

6. TAAL LINE BEAT COUNT
   Do NOT change the beat count of taal (non-adlib) lines.
   Each must remain exactly {beats_per_line} beats.

7. NO INVENTED NOTES
   Do not add notes that are not present in the input data.

RAW DATA:
{json.dumps(raw_lines, indent=2)}

Return ONLY a valid JSON object with ONE key "cleaned_lines" containing the
cleaned array with the same structure as the input.  No markdown, no explanation.
"""

print("\nCalling AI for cleanup…")
cleaned = call_gemini(prompt)
if not cleaned:
    print("Gemini failed — trying Mistral…")
    cleaned = call_mistral(prompt)

if cleaned:
    if isinstance(cleaned, dict) and "cleaned_lines" in cleaned:
        cleaned = cleaned["cleaned_lines"]
    new_lines = []
    for cl in cleaned:
        is_adlib = cl.get("adlib", False)
        notes_cl = cl.get("notes", [])
        octs_cl  = cl.get("octaves", [])
        new_lines.append({
            "section":         cl.get("section", ""),
            "line_instrument": cl.get("line_instrument", "harmonium"),
            "percussion":      "mute" if is_adlib else "tabla",
            "line_volume":     1.0,
            "adlib":           is_adlib,
            "legato":          is_adlib,
            "line_bpm":        None,
            "taal_key":        str(beats_per_line),
            "beats":           cl.get("beats", len(notes_cl)),
            "lyrics":          [[""] for _ in notes_cl],
            "notes":           notes_cl,
            "octaves":         octs_cl,
            "meend":           [[False] * len(b) for b in notes_cl],
        })
    final_output["lines"] = new_lines
    with open("swar_laya_output.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print("AI-cleaned transcription saved → swar_laya_output.json")
else:
    print("AI cleanup failed.  Load precision_output.json and edit manually.")
