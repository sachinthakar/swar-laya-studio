#!/usr/bin/env python3
"""
mp3_to_swarlaya.py  -  Swar-Laya Studio JSON generator from MP3

Analyses an MP3 file using pitch detection (HPS algorithm) and outputs
a JSON file that can be loaded directly into Swar-Laya Studio v3.

Requirements:
    pip install miniaudio numpy scipy

Usage:
    python mp3_to_swarlaya.py song.mp3
    python mp3_to_swarlaya.py song.mp3 --scale G#
    python mp3_to_swarlaya.py song.mp3 --scale G# --taal Keherwa --bpm 75
    python mp3_to_swarlaya.py song.mp3 --scale G# --melody-instrument harmonium
    python mp3_to_swarlaya.py song.mp3 --scale C  --taal Teentaal --bpm 120 --verbose

Arguments:
    mp3_file                  Path to the MP3 file (required)

  Pitch / Scale:
    --scale  NOTE             Assume Sa (tonic) is this note: C C# D D# E F F# G G# A A# B
                              All detected pitches are mapped relative to this tonic.
                              Optionally add octave: G#4 (default octave = 4).
                              This is the note the tanpura drone is tuned to.
                              Examples: --scale G#  --scale C  --scale D#4
    --sa     HZ               Assume Sa (tonic) is this frequency in Hz (alternative to --scale)
                              e.g. --sa 415.3  is the same as --scale G#

  Tempo / Taal:
    --taal   NAME             Taal name (sets beats and taal_key automatically)
                              Keherwa(8)  Bhajani(8)  Dadra(6)  Rupak(7)
                              Ektaal(12)  Teentaal(16)  Jhaptaal(10)
    --beats  N                Beats per cycle override (4/6/7/8/10/12/16)
    --bpm    N                Tempo in BPM (auto-detected if not specified)

  Instruments:
    --melody-instrument NAME  Instrument for Mukhda/Antara sections
                              harmonium (default) | violin | flute | sitar | piano
    --intro-instrument  NAME  Instrument for Intro/Alaap sections
                              violin (default)    | harmonium | flute | sitar | piano

  Output:
    --title  TEXT             Song title (defaults to filename)
    --output PATH             Output JSON path (defaults to <mp3_name>.json)
    --max-lines N             Maximum lines in JSON (default: 50)
    --dedup                   Merge consecutive identical cycles into one line
    --auto-meend              Enable heuristic meend (glide) detection.
                              Off by default -- auto-meend sounds bad on most songs;
                              set meend manually per note in the studio after listening.
    --fmin   HZ               Minimum pitch frequency for the HPS detector.
                              Auto-computed as base_sa * 0.45 when --scale is given.
                              Override only if the auto value cuts real melody notes.
    --song-start NOTE         Lowest swara expected in the metered song section.
                              Auto-computes the main HPS fmin as just below that swara.
                              More musical than --fmin HZ.  Requires --scale or --sa.
                              Example: --song-start Ga  (voice never goes below Ga)
    --intro-fmin HZ           Minimum pitch for the intro/alaap HPS re-analysis (Hz).
                              Used to suppress tanpura drone during soft alaap.
    --intro-start NOTE        Lowest swara expected in the intro/alaap.
                              Auto-computes intro fmin as just below that swara.
                              More musical than --intro-fmin HZ.
                              Example: --intro-start Pa  (alaap starts around Pa)
    --verbose                 Print cycle-by-cycle breakdown
"""

import sys
import os
import re
import json
import argparse
import math
from collections import Counter

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    sys.exit("Error: numpy not installed. Run:  pip install numpy")

try:
    import miniaudio
except ImportError:
    sys.exit("Error: miniaudio not installed. Run:  pip install miniaudio")

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found - BPM detection will be less accurate.")
    print("         Install with:  pip install scipy")


# ===========================================================================
# CONSTANTS
# ===========================================================================

# Swaras in semitone order - must match Swar-Laya Studio v3's SwaraMap exactly
SWARA_NAMES = ['Sa', 're', 'Re', 'ga', 'Ga', 'Ma', 'ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni']
SWARA_IDX   = {name: i for i, name in enumerate(SWARA_NAMES)}
SWARA_RATIO = {name: 2 ** (i / 12.0) for i, name in enumerate(SWARA_NAMES)}

MAX_PITCH_ERROR_CENTS = 70.0   # reject if pitch is > 70c from nearest swara

# ---------------------------------------------------------------------------
# Scale note names -> playing frequency (baseSa = scale_value * 2) at octave 4
# e.g. G# -> 415.30 Hz (G#4)
# ---------------------------------------------------------------------------
_C4 = 261.6256
_CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_FLAT_MAP  = {'Db':'C#', 'Eb':'D#', 'Fb':'E', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#', 'Cb':'B'}

# baseSa Hz for each chromatic note at octave 4
SCALE_NOTE_HZ = {note: _C4 * (2 ** (i / 12.0)) for i, note in enumerate(_CHROMATIC)}

# Scale value lookup: baseSa_hz -> nearest "scale" string for the JSON
# In Swar-Laya Studio: scale_value = baseSa / 2
# We pre-build a table over 2 octaves (C3-B4)
def _build_scale_table():
    C3 = 130.8128
    out = {}
    for oct_n in [0, 1]:
        for semi, note in enumerate(_CHROMATIC):
            hz = C3 * (2 ** (oct_n + semi / 12.0))
            out[round(hz / 2, 2)] = f"{note}{3 + oct_n}"  # "G#3", "C4", ...
    return out
SCALE_TABLE = _build_scale_table()   # {130.81: "C3", 207.65: "G#3", ...}

# ---------------------------------------------------------------------------
# Taal name -> (taal_key_for_json, beats_per_cycle)
# taal_key must match the keys in Swar-Laya Studio v3's TaalBeats lookup
# ---------------------------------------------------------------------------
TAAL_MAP = {
    'keherwa':   ('8',         8),
    'keherwaa':  ('8',         8),
    'bhajani':   ('8_bhajani', 8),
    'dadra':     ('6',         6),
    'rupak':     ('7',         7),
    'jhaptaal':  ('10',       10),   # Note: may need studio support
    'jhampa':    ('10',       10),
    'ektaal':    ('12',       12),
    'teentaal':  ('16',       16),
    'tintal':    ('16',       16),
    'tintaal':   ('16',       16),
    'teen':      ('16',       16),
    'char':      ('4',         4),
}

VALID_INSTRUMENTS = ['harmonium', 'violin', 'flute', 'sitar', 'piano']


# ===========================================================================
# STEP 1 - Load MP3
# ===========================================================================

def load_mp3(path):
    """Decode MP3 to mono float32.  Returns (audio_ndarray, sample_rate)."""
    mp3   = miniaudio.mp3_read_file_f32(path)
    audio = np.array(mp3.samples, dtype=np.float32)
    if mp3.nchannels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    return audio, int(mp3.sample_rate)


# ===========================================================================
# STEP 2 - Pitch detection (Harmonic Product Spectrum)
# ===========================================================================

def _hps(frame, sr, fmin=150.0, fmax=1200.0, n_harmonics=5):
    """
    Estimate fundamental frequency via Harmonic Product Spectrum.
    Returns Hz float, or None when no confident pitch is found.

    fmin: lower bound for pitch search.  Set to ~0.52*base_sa to exclude the
          tanpura drone (which plays Sa one octave below the melody).  The
          caller (compute_pitch_frames) supplies the right value once base_sa
          is known.
    """
    N = len(frame)
    if N < 512:
        return None
    fft_n    = N * 4
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(N), n=fft_n))
    freqs    = np.fft.rfftfreq(fft_n, 1.0 / sr)

    hps = spectrum.copy()
    for h in range(2, n_harmonics + 1):
        dec = spectrum[::h]
        hps[:len(dec)] *= dec

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return None
    hps_m = hps[mask]
    peak  = int(np.argmax(hps_m))
    # Confidence: peak must be at least 5.5x the mean.
    # The tanpura drone creates a broad low-frequency hump; a higher threshold
    # ensures only clear melody notes (with strong harmonic alignment) survive,
    # even when fmin is low enough to admit the melody's lower-octave range.
    if hps_m[peak] < np.mean(hps_m) * 5.5:
        return None
    return float(freqs[mask][peak])


def compute_pitch_frames(audio, sr, hop_ms=40, frame_ms=60, fmin=150.0):
    """
    Run pitch detection every hop_ms ms using a frame_ms window.
    Returns (times_ndarray, pitches_list)  where pitches may contain None.

    fmin: passed straight to _hps.  Supply base_sa * 0.52 to exclude the
          tanpura drone octave (Sa-1) while keeping all melody notes.
    """
    hop  = max(1,   int(sr * hop_ms   / 1000))
    flen = max(512, int(sr * frame_ms / 1000))
    times, pitches = [], []
    for i in range(0, len(audio) - flen, hop):
        times.append(i / sr)
        pitches.append(_hps(audio[i:i + flen], sr, fmin=fmin))
    return np.array(times), pitches


# ===========================================================================
# STEP 3 - Auto-detect Sa (tonic)
# ===========================================================================

def detect_sa(pitches, verbose=False):
    """
    Estimate Sa by finding the most common sustained pitch in 200-600 Hz.
    The tanpura drone continuously plays Sa, so it dominates the pitch histogram.
    Returns baseSa in Hz (the playing frequency, i.e. what the studio uses).
    """
    valid = [f for f in pitches if f is not None and 180 < f < 650]
    if not valid:
        print("  Warning: cannot detect Sa automatically; defaulting to A4=440 Hz")
        return 440.0

    # Round to nearest semitone (relative to A4=440)
    to_semi = lambda f: round(12 * math.log2(f / 440.0))
    counts  = Counter(to_semi(f) for f in valid)

    if verbose:
        print("  Pitch histogram (top 6 candidates):")
        for semi, cnt in counts.most_common(6):
            hz = 440.0 * (2 ** (semi / 12.0))
            while hz > 600: hz /= 2
            while hz < 180: hz *= 2
            print(f"    {hz:6.1f} Hz  ({cnt:4d} frames)")

    best_semi = counts.most_common(1)[0][0]
    sa_hz     = 440.0 * (2 ** (best_semi / 12.0))
    # Normalise to 200-500 Hz range (typical singing Sa)
    while sa_hz > 500: sa_hz /= 2
    while sa_hz < 200: sa_hz *= 2
    return sa_hz


def parse_scale_arg(scale_str):
    """
    Parse a scale argument like 'G#', 'C', 'D#4', 'Bb3'
    and return the baseSa playing frequency in Hz.
    """
    scale_str = scale_str.strip()
    m = re.match(r'^([A-Ga-g][#bB]?)(\d?)$', scale_str)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid scale '{scale_str}'. Use note names like C, C#, D, G#, A#")

    note_raw = m.group(1)
    octave   = int(m.group(2)) if m.group(2) else None

    # Normalise to uppercase + # notation
    note = note_raw[0].upper() + note_raw[1:].replace('b', '#') \
           if len(note_raw) == 1 else note_raw[0].upper() + note_raw[1]
    note = _FLAT_MAP.get(note, note)

    if note not in SCALE_NOTE_HZ:
        raise argparse.ArgumentTypeError(
            f"Unknown note '{note}'. Use: C C# D D# E F F# G G# A A# B")

    base_hz = SCALE_NOTE_HZ[note]   # baseSa at octave 4

    if octave is not None:
        # Shift from default octave-4 to requested octave
        base_hz *= 2 ** (octave - 4)

    return base_hz


def sa_to_scale_str(base_sa):
    """Return the JSON scale string (= baseSa/2, snapped to nearest chromatic note)."""
    target    = base_sa / 2.0
    best_val  = min(SCALE_TABLE.keys(), key=lambda v: abs(v - target))
    return str(best_val)


# ===========================================================================
# STEP 4 - BPM detection
# ===========================================================================

def detect_bpm(audio, sr, bpm_low=55, bpm_high=160):
    """
    Detect tempo via onset-strength envelope + beat-grid scoring.
    Also checks half / double of the best candidate to avoid octave errors.
    Returns BPM as int.
    """
    hop  = max(1,  int(sr * 0.010))   # 10 ms hop
    flen = max(64, int(sr * 0.030))   # 30 ms frame

    n_frames = (len(audio) - flen) // hop
    env = np.array([np.sqrt(np.mean(audio[i*hop: i*hop+flen]**2))
                    for i in range(n_frames)], dtype=np.float64)

    # High-pass to emphasise onsets
    if HAS_SCIPY:
        b, a    = butter(2, 0.08, btype='high')
        env_hp  = np.maximum(filtfilt(b, a, env), 0)
    else:
        env_hp  = np.maximum(np.diff(env, prepend=env[0]), 0)

    fps = sr / hop   # envelope samples per second

    def grid_score(bpm_cand):
        bf = fps * 60.0 / bpm_cand
        s, pos = 0.0, 0.0
        while pos < len(env_hp):
            s  += env_hp[int(round(pos)) % len(env_hp)]
            pos += bf
        return s

    scores = {b: grid_score(b) for b in range(bpm_low, bpm_high + 1)}
    best   = max(scores, key=scores.__getitem__)

    # Resolve half/double ambiguity
    half   = best // 2
    double = best * 2
    if bpm_low <= half <= bpm_high and grid_score(half) > grid_score(best) * 0.80:
        best = half
    elif bpm_low <= double <= bpm_high and grid_score(double) > grid_score(best) * 1.30:
        best = double

    return int(best)


# ===========================================================================
# STEP 5 - Frequency -> Swara mapping
# ===========================================================================

def freq_to_swara(freq, base_sa):
    """
    Map a frequency to (swara_name, octave) relative to base_sa.
      octave  0 = middle (Sa at base_sa)
      octave -1 = lower  (Sa at base_sa/2)
      octave +1 = upper  (Sa at base_sa*2)
    Returns ('-', 0) when freq is None or too far from any swara.
    """
    if freq is None or freq < 50:
        return '-', 0

    f, octave = float(freq), 0
    lo = base_sa * 2 ** (-0.5 / 12)
    hi = base_sa * 2 ** (11.5 / 12)

    while f < lo:
        f      *= 2
        octave -= 1
    while f > hi:
        f      /= 2
        octave += 1

    # Clamp octave to range the studio can display (-2 to +2)
    octave = max(-2, min(2, octave))

    best_name, best_err = '-', MAX_PITCH_ERROR_CENTS
    for name, ratio in SWARA_RATIO.items():
        cents = abs(1200.0 * math.log2(f / (base_sa * ratio)))
        if cents < best_err:
            best_err  = cents
            best_name = name

    return best_name, octave


# ===========================================================================
# STEP 6 - Beat-synchronous note extraction
# ===========================================================================

def extract_beats(times, pitches, base_sa, bpm, total_duration, t_start=0.0):
    """
    Assign the dominant swara to each beat of the song.

    t_start: skip this many seconds at the beginning (free-tempo intro already
             handled by extract_free_section). Phase search only runs over
             [t_start, total_duration], giving a much cleaner alignment for
             the metered section.

    Phase alignment: tries 16 starting offsets and picks the one that gives
    the most consistent (single-note) beats.

    Returns list of dicts: [{'time': float, 'note': str, 'octave': int}, ...]
    """
    beat_sec = 60.0 / bpm
    times_a  = np.array(times)

    def window_swara(t0):
        mask  = (times_a >= t0) & (times_a < t0 + beat_sec)
        wp    = [pitches[i] for i in np.where(mask)[0] if pitches[i] is not None]
        if not wp:
            return '-', 0
        sw_oc = [freq_to_swara(p, base_sa) for p in wp]
        valid  = [(s, o) for s, o in sw_oc if s != '-']
        if not valid:
            return '-', 0
        dom   = Counter(s for s, o in valid).most_common(1)[0][0]
        dom_octaves    = [o for s, o in valid if s == dom]
        dom_oct_ctr    = Counter(dom_octaves)
        dom_o          = dom_oct_ctr.most_common(1)[0][0]
        # Prefer middle octave (0) over lower octaves if the melody is
        # playing in the middle register even when the tanpura drone is louder.
        # Rule: if >=20% of frames for the dominant note are at oct=0
        # and the current winner is a lower octave, promote to oct=0.
        if dom_o < 0:
            oct0_frac = dom_oct_ctr.get(0, 0) / len(dom_octaves)
            if oct0_frac >= 0.20:
                dom_o = 0
        return dom, dom_o

    # Phase search -- only over the metered region [t_start, total_duration]
    best_ph, best_sc = t_start, -1.0
    for k in range(16):
        ph = t_start + k * beat_sec / 16
        sc, n = 0.0, 0
        t = ph
        while t + beat_sec < total_duration:
            mask = (times_a >= t) & (times_a < t + beat_sec)
            wp   = [pitches[i] for i in np.where(mask)[0] if pitches[i] is not None]
            if wp:
                sw = [freq_to_swara(p, base_sa)[0] for p in wp]
                c  = Counter(sw)
                sc += c.most_common(1)[0][1] / len(sw)
                n  += 1
            t += beat_sec
        avg = sc / n if n else 0
        if avg > best_sc:
            best_sc, best_ph = avg, ph

    # Extract beats starting from the best phase in the metered region
    beats, t = [], best_ph
    while t < total_duration - beat_sec * 0.4:
        note, oct_ = window_swara(t)
        beats.append({'time': round(t, 3), 'note': note, 'octave': oct_})
        t += beat_sec
    return beats


# ===========================================================================
# STEP 7 - Group beats into cycles
# ===========================================================================

def beats_to_cycles(beats, bpc):
    """Split beat list into complete cycles of bpc beats each."""
    cycles = []
    for i in range(0, len(beats) - bpc + 1, bpc):
        cycle = [(b['note'], b['octave']) for b in beats[i: i + bpc]]
        cycles.append({'data': cycle, 'start_time': beats[i]['time']})
    return cycles


# ===========================================================================
# STEP 7b - Free-tempo (alaap / intro) note extraction
# ===========================================================================

def extract_free_section(times_a, pitches, base_sa, t_start, t_end,
                         win_sec=0.08, hop_sec=0.02, min_dur_sec=0.15,
                         min_octave=-1):
    """
    Pitch analysis for free-tempo sections (alaap intro).

    Instead of a fixed beat grid, this slides a short window (win_sec wide)
    every hop_sec seconds over [t_start, t_end], finds the dominant swara in
    each window, then removes consecutive duplicates that are shorter than
    min_dur_sec.  The result is a list of (swara, octave) pairs, one per
    distinct sustained note event -- independent of any BPM assumption.

    Parameters
    ----------
    win_sec     Width of each analysis window (default 80 ms).
                Shorter than the metered-section window to capture the rapid,
                subtle ornaments (meend, gamak) of flute and sitar alaap.
    hop_sec     Step between windows (default 20 ms).
    min_dur_sec Minimum duration a note must be sustained to survive
                the dedup pass (default 150 ms).  Stricter than the
                metered-section gate to suppress tanpura hum and breath-noise
                blips that lack a true sustain.
    min_octave  Lowest octave that may appear in the output (default -1).
                Rejects ghosting from the tanpura drone or room resonance,
                which can fool the HPS detector into returning octave -2
                when fmin is set just above the tanpura fundamental.
                Use -2 to allow very-low mandra notes; use 0 for alaap
                that stays in the middle/upper octave only.
    """
    note_seq = []   # [(swara, octave, window_start_time), ...]
    t = t_start
    while t + win_sec <= t_end:
        mask = (times_a >= t) & (times_a < t + win_sec)
        wp   = [pitches[i] for i in np.where(mask)[0] if pitches[i] is not None]
        if wp:
            sw_oc = [freq_to_swara(p, base_sa) for p in wp]
            # Filter: skip silent frames, and enforce the octave floor
            valid = [(s, o) for s, o in sw_oc if s != '-' and o >= min_octave]
            if valid:
                dom    = Counter(s for s, o in valid).most_common(1)[0][0]
                dom_oc = Counter(o for s, o in valid if s == dom).most_common(1)[0][0]
                note_seq.append((dom, dom_oc, t))
        t += hop_sec

    if not note_seq:
        return []

    # Consecutive-duplicate removal with minimum-duration gate
    events = []
    prev_sw, prev_oc, seg_start = note_seq[0]
    for sw, oc, t in note_seq[1:] + [(None, None, t_end)]:
        if sw != prev_sw:
            if (t - seg_start) >= min_dur_sec:
                events.append((prev_sw, prev_oc))
            prev_sw, prev_oc, seg_start = sw, oc, t

    return events   # list of (swara_name, octave)


def notes_to_cycles(notes, bpc):
    """
    Pack a flat list of (swara, octave) note events into cycles of bpc notes.
    Incomplete final cycles are padded with ('-', 0).
    """
    cycles = []
    for i in range(0, len(notes), bpc):
        chunk = list(notes[i: i + bpc])
        if not chunk:
            break
        while len(chunk) < bpc:
            chunk.append(('-', 0))
        cycles.append({'data': chunk, 'start_time': 0.0})
    return cycles


# ===========================================================================
# STEP 8 - Section identification
# ===========================================================================

def _note_vector(cycle_data):
    """12-element note histogram (normalised) for cosine similarity."""
    v = np.zeros(12)
    for note, _ in cycle_data:
        if note in SWARA_IDX:
            v[SWARA_IDX[note]] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


def _cosine(va, vb):
    d = float(np.dot(va, vb))
    return d / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)


def identify_sections(cycles, intro_frac=0.15, sim_thresh=0.80, intro_cycles=None):
    """
    Assign a section label to each cycle:
      'Intro'            - cycles before the first mukhda occurrence
                           (or first intro_cycles cycles if specified,
                            or first intro_frac fraction as fallback)
      'Mukhda'           - most frequently recurring melodic pattern
      'Mukhda (Repeat)'  - later occurrences of the mukhda pattern
      'Antara'           - other melodic sections
      'Antara 2', ...    - additional distinct antara patterns

    Uses exact note fingerprints first; falls back to cosine similarity.
    intro_cycles: if given, forces exactly that many cycles to be labelled Intro.
    """
    n = len(cycles)
    if n == 0:
        return cycles
    fps          = [tuple(note for note, _ in c['data']) for c in cycles]
    fp_counts    = Counter(fps)

    # Most common fingerprint = mukhda candidate (must appear >= 2 times)
    top_fp, top_cnt = fp_counts.most_common(1)[0]
    mukhda_fp        = top_fp if top_cnt >= 2 else None

    # Intro end: explicit override > content-based > fixed-fraction fallback
    if intro_cycles is not None:
        intro_end = max(0, min(int(intro_cycles), n))
    elif mukhda_fp is not None:
        try:
            intro_end = fps.index(mukhda_fp)  # 0 = first cycle IS mukhda (no intro)
        except ValueError:
            intro_end = max(1, int(n * intro_frac))
    else:
        intro_end = max(1, int(n * intro_frac))

    # If no exact repeat, use cosine cluster
    if mukhda_fp is None and n > 3:
        vecs     = [_note_vector(c['data']) for c in cycles]
        avg_sims = [np.mean([_cosine(vecs[i], vecs[j])
                              for j in range(n) if j != i])
                    for i in range(n)]
        ref_idx     = int(np.argmax(avg_sims))
        ref_vec     = vecs[ref_idx]
        is_mukhda   = lambda i: _cosine(vecs[i], ref_vec) >= sim_thresh
    else:
        is_mukhda   = lambda i: fps[i] == mukhda_fp

    antara_map  = {}   # fp -> 'Antara' or 'Antara 2' etc.
    antara_cnt  = [1]
    mukhda_seen = [0]

    for i, cy in enumerate(cycles):
        fp = fps[i]
        if i < intro_end:
            label = 'Intro'
        elif is_mukhda(i):
            if mukhda_seen[0] == 0:
                label = 'Mukhda'
            else:
                label = f'Mukhda (Repeat {mukhda_seen[0]})'
            mukhda_seen[0] += 1
        else:
            if fp not in antara_map:
                suffix = '' if antara_cnt[0] == 1 else f' {antara_cnt[0]}'
                antara_map[fp] = f'Antara{suffix}'
                antara_cnt[0] += 1
            label = antara_map[fp]
        cycles[i]['section'] = label

    return cycles


def deduplicate_consecutive(cycles):
    """
    Merge consecutive cycles that share the exact same note fingerprint into
    one entry. Keeps the first occurrence of each consecutive run.
    (Only used when --dedup flag is set.)
    """
    if not cycles:
        return []
    result = [cycles[0]]
    for cy in cycles[1:]:
        same_fp = (tuple(n for n, _ in cy['data']) ==
                   tuple(n for n, _ in result[-1]['data']))
        if not same_fp:
            result.append(cy)
    return result


# ===========================================================================
# STEP 9 - Meend detection
# ===========================================================================

def compute_meend(note_seq, auto_meend=False):
    """
    Return a list of meend (glide) flags for each beat.

    auto_meend=False (default):
        All flags are False.  Meend should be added manually in the studio
        after listening, because the heuristic sounds unnatural on most songs.

    auto_meend=True (--auto-meend flag):
        Mark meend=True when the previous note is lower by 1-4 semitones
        (ascending stepwise motion).  Use only if the auto-detected glides
        actually sound correct for the song.
    """
    result = [False] * len(note_seq)
    if not auto_meend:
        return result
    for i in range(1, len(note_seq)):
        prev, curr = note_seq[i - 1], note_seq[i]
        if prev == '-' or curr == '-':
            continue
        pi = SWARA_IDX.get(prev, -1)
        ci = SWARA_IDX.get(curr, -1)
        if 0 <= pi and 0 <= ci and 1 <= (ci - pi) <= 4:
            result[i] = True
    return result


# ===========================================================================
# STEP 10 - Build JSON
# ===========================================================================

def build_json(cycles, title, scale_str, bpm,
               taal_key, bpc,
               melody_inst, intro_inst,
               auto_meend=False):
    """
    Convert labelled cycles into a Swar-Laya Studio v3 compatible JSON dict.

    auto_meend: if True, heuristic meend flags are applied on ascending
                stepwise motion.  Default False (all meend flags are off).
    """
    lines       = []
    intro_done  = False   # tabla stays off through intro
    mukhda_seen = [False]

    for cy in cycles:
        label    = cy.get('section', 'Mukhda')
        is_intro = 'Intro' in label

        # Instrument selection
        instrument = intro_inst if is_intro else melody_inst

        # Tabla: off during intro; off for first Mukhda line (traditional alaap end)
        if is_intro:
            tabla_mute = True
        elif label == 'Mukhda' and not mukhda_seen[0]:
            tabla_mute     = True     # tabla begins on the response, not the call
            mukhda_seen[0] = True
        else:
            tabla_mute = False

        # Notes, octaves, meend
        notes   = [n for n, _ in cy['data']]
        octaves = [o for _, o in cy['data']]
        meend   = compute_meend(notes, auto_meend=auto_meend)

        # Pad / trim to exact bpc
        while len(notes) < bpc:
            notes.append('-'); octaves.append(0); meend.append(False)
        notes   = notes[:bpc]
        octaves = octaves[:bpc]
        meend   = meend[:bpc]

        lines.append({
            'section':         label,
            'line_instrument': instrument,
            'tabla_mute':      tabla_mute,
            'line_volume':     0.85 if is_intro else 1.0,
            'taal_key':        taal_key,
            'beats':           str(bpc),
            'lyrics':          ['-'] * bpc,
            'notes':           [[n] for n in notes],
            'octaves':         [[o] for o in octaves],
            'meend':           [[m] for m in meend],
        })

    return {'title': title, 'scale': scale_str, 'bpm': bpm, 'lines': lines}


# ===========================================================================
# MAIN
# ===========================================================================

# ===========================================================================
# VERIFY MODE  --  compare an existing JSON against the source MP3
# ===========================================================================

def _semitone_distance(note_a, note_b):
    """Return the smallest semitone distance between two swara names (0-6)."""
    ia = SWARA_IDX.get(note_a, -1)
    ib = SWARA_IDX.get(note_b, -1)
    if ia < 0 or ib < 0:
        return 99
    d = abs(ia - ib)
    return min(d, 12 - d)   # wrap around the octave


def _plot_verification(times_a, pitches, results, beat_sec, base_sa, title, json_path):
    """Save a two-panel PNG: (1) pitch contour + JSON dots, (2) beat accuracy bar."""
    try:
        import matplotlib
        matplotlib.use('Agg')          # headless -- works without a display
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [plot skipped -- matplotlib not installed:  pip install matplotlib]")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8),
                                    gridspec_kw={'height_ratios': [4, 1]})
    fig.subplots_adjust(hspace=0.35)

    # ── Panel 1: MP3 pitch contour + JSON note dots ──────────────────────
    ax1.set_title(f"Pitch comparison -- {title}", fontsize=12, pad=6)

    mp3_t, mp3_s = [], []
    for t, p in zip(times_a, pitches):
        if p is None:
            continue
        ratio = p / base_sa
        while ratio < 2 ** (-0.5 / 12):
            ratio *= 2
        while ratio > 2 ** (11.5 / 12):
            ratio /= 2
        semi = 12.0 * math.log2(ratio)
        if -1.0 <= semi <= 12.5:
            mp3_t.append(float(t))
            mp3_s.append(semi)

    ax1.plot(mp3_t, mp3_s, color='#8888dd', alpha=0.55, lw=0.7,
             label='MP3 pitch (semitones from Sa)')

    # JSON note dots coloured by match quality
    for r in results:
        if r['json_note'] == '-':
            continue
        semi = SWARA_IDX.get(r['json_note'], None)
        if semi is None:
            continue
        if r['exact']:
            col, zord = '#22cc66', 6
        elif r['close']:
            col, zord = '#ffaa00', 5
        else:
            col, zord = '#ff3333', 5
        ax1.scatter([r['time']], [semi], color=col, s=55, zorder=zord,
                    edgecolors='white', linewidths=0.4)
        ax1.annotate(r['json_note'], (r['time'], semi),
                     textcoords='offset points', xytext=(0, 5),
                     fontsize=6.5, ha='center', color=col)

    ax1.set_yticks(range(12))
    ax1.set_yticklabels(SWARA_NAMES, fontsize=8)
    ax1.set_ylabel('Swara (semitone above Sa)')
    ax1.grid(axis='y', alpha=0.25, lw=0.5)
    ax1.set_xlim(0, float(times_a[-1]) if len(times_a) else 1)

    patches = [
        mpatches.Patch(color='#22cc66', label='Exact match'),
        mpatches.Patch(color='#ffaa00', label='Close (±1 semitone)'),
        mpatches.Patch(color='#ff3333', label='Mismatch'),
        mpatches.Patch(color='#8888dd', label='MP3 pitch'),
    ]
    ax1.legend(handles=patches, fontsize=8, loc='upper right')

    # Section dividers
    prev_sec = None
    for r in results:
        if r['section'] != prev_sec:
            ax1.axvline(r['time'], color='#aaaaaa', lw=0.7, ls='--')
            ax1.text(r['time'] + 0.1, 11.5, r['section'],
                     fontsize=7, color='#666666', va='top')
            prev_sec = r['section']

    # ── Panel 2: per-beat bar ─────────────────────────────────────────────
    ax2.set_title('Beat-by-beat: green=exact  orange=close(±1st)  red=wrong  gray=rest',
                  fontsize=8, pad=4)
    for i, r in enumerate(results):
        if r['json_note'] == '-':
            col = '#cccccc'
        elif r['exact']:
            col = '#22cc66'
        elif r['close']:
            col = '#ffaa00'
        else:
            col = '#ff3333'
        ax2.bar(i, 1.0, color=col, edgecolor='white', linewidth=0.2)

    ax2.set_xlim(-0.5, len(results) - 0.5)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel('Beat index (all lines concatenated)')

    plot_path = os.path.splitext(json_path)[0] + '_verify.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved -> {plot_path}")


def run_verify(args):
    """
    Verification mode: compare an existing JSON against the source MP3.

    Algorithm
    ---------
    1. Load JSON  ->  extract base_sa, bpm, and the per-beat note sequence
       with estimated timestamps (beat 0 of line 0 = t=0, each subsequent
       beat advances by beat_sec = 60/bpm).
    2. Run HPS pitch detection on the MP3 (same fmin heuristic as generation).
    3. For each JSON beat at time t, sample the MP3 pitch in the window
       [t − 0.25·beat, t + 0.75·beat].  Take the median of confident frames.
    4. Compare JSON note to detected MP3 note:
         exact  -- same swara name
         close  -- within ±1 semitone  (e.g. komal/shuddha confusion)
         wrong  -- ≥ 2 semitones away
    5. Print a per-beat table and overall/per-section accuracy.
    6. Optionally save a matplotlib PNG (--plot).

    Timing note: beat-aligned sampling works well for the metered (tabla)
    section.  For free-tempo alaap the timestamps are approximate; the
    window overlap (±0.25 beat) absorbs small drifts but rubato passages
    will still show some mismatches even when the notes are correct.
    """
    json_path = args.verify
    mp3_path  = args.mp3_file

    if not os.path.isfile(mp3_path):
        sys.exit(f"Error: MP3 not found: {mp3_path}")
    if not os.path.isfile(json_path):
        sys.exit(f"Error: JSON not found: {json_path}")

    # 1. Load JSON ────────────────────────────────────────────────────────
    try:
        with open(json_path, encoding='utf-8') as f:
            jdata = json.load(f)
    except Exception as e:
        sys.exit(f"Error reading JSON: {e}")

    try:
        base_sa = float(jdata['scale']) * 2
    except (KeyError, ValueError):
        sys.exit("JSON missing valid 'scale' field.")

    bpm      = float(jdata.get('bpm', 90))
    beat_sec = 60.0 / bpm
    title    = jdata.get('title', os.path.basename(json_path))

    sep = '=' * 68
    print(sep)
    print(f"  Verify: {title}")
    print(f"  MP3   : {mp3_path}")
    print(f"  JSON  : {json_path}")
    print(f"  Sa = {base_sa:.2f} Hz   BPM = {bpm:.0f}   beat = {beat_sec:.3f}s")
    print(sep)

    # 2. Pitch detection on MP3 ──────────────────────────────────────────
    fmin = args.fmin if (hasattr(args, 'fmin') and args.fmin) else base_sa * 0.80
    print(f"  Running HPS pitch analysis  (fmin={fmin:.1f} Hz)...")
    audio, sr = load_mp3(mp3_path)
    duration  = len(audio) / sr
    times, pitches = compute_pitch_frames(audio, sr, fmin=fmin)
    times_a = np.array(times)
    valid_pct = 100 * sum(1 for p in pitches if p is not None) / max(len(pitches), 1)
    print(f"  {len(pitches)} frames, {valid_pct:.0f}% with confident pitch")
    print()

    # 3. Build beat timeline from JSON lines ─────────────────────────────
    beat_timeline = []
    t_cursor = 0.0
    for li, line in enumerate(jdata.get('lines', [])):
        n_beats  = int(line.get('beats', 8))
        notes_a  = line.get('notes',  [])
        octs_a   = line.get('octaves', [])
        section  = line.get('section', f'Line {li+1}')
        for bi in range(n_beats):
            note = notes_a[bi][0] if bi < len(notes_a) and notes_a[bi] else '-'
            oct_ = octs_a[bi][0]  if bi < len(octs_a)  and octs_a[bi]  else 0
            beat_timeline.append({
                'time':      t_cursor + bi * beat_sec,
                'line_no':   li + 1,
                'beat_no':   bi + 1,
                'section':   section,
                'json_note': note,
                'json_oct':  oct_,
            })
        t_cursor += n_beats * beat_sec

    # 4. Per-beat comparison ──────────────────────────────────────────────
    results = []
    for b in beat_timeline:
        t0   = b['time']
        t_lo = max(0.0, t0 - beat_sec * 0.25)
        t_hi = min(duration, t0 + beat_sec * 0.75)
        mask = (times_a >= t_lo) & (times_a < t_hi)
        wp   = [pitches[i] for i in np.where(mask)[0] if pitches[i] is not None]

        if len(wp) >= 3:
            mp3_note, mp3_oct = freq_to_swara(float(np.median(wp)), base_sa)
        else:
            mp3_note, mp3_oct = '?', 0     # too few confident frames in window

        dist  = _semitone_distance(b['json_note'], mp3_note)
        exact = (b['json_note'] == mp3_note) and b['json_note'] != '-'
        close = (not exact) and (dist == 1)  and b['json_note'] != '-'
        wrong = (not exact) and (not close)  and b['json_note'] != '-'

        results.append({
            **b,
            'mp3_note': mp3_note,
            'mp3_oct':  mp3_oct,
            'dist':     dist,
            'exact':    exact,
            'close':    close,
            'wrong':    wrong,
        })

    # 5. Print per-line report ────────────────────────────────────────────
    def _note_str(note, oct_):
        if note in ('-', '?'):
            return note
        suffix = '↑' if oct_ > 0 else ('↓' if oct_ < 0 else '')
        return note + suffix

    prev_line = None
    print(f"  {'Ln':>3} {'Bt':>3}  {'Section':<22}  "
          f"{'JSON':>6}  {'MP3':>6}  {'dist':>5}  Result")
    print('  ' + '─' * 64)

    for r in results:
        if r['line_no'] != prev_line:
            if prev_line is not None:
                print()
            prev_line = r['line_no']

        if r['json_note'] == '-':
            sym = '·'
        elif r['exact']:
            sym = '✓'
        elif r['close']:
            sym = '~'      # close miss
        else:
            sym = '✗'

        dist_s = f"{r['dist']:>2}st" if r['json_note'] != '-' else '   '
        print(f"  {r['line_no']:>3} {r['beat_no']:>3}  {r['section']:<22}  "
              f"{_note_str(r['json_note'], r['json_oct']):>6}  "
              f"{_note_str(r['mp3_note'],  r['mp3_oct']):>6}  "
              f"{dist_s}  {sym}")

    # 6. Summary statistics ───────────────────────────────────────────────
    scored  = [r for r in results if r['json_note'] != '-']
    n_exact = sum(1 for r in scored if r['exact'])
    n_close = sum(1 for r in scored if r['close'])
    n_wrong = sum(1 for r in scored if r['wrong'])
    n_total = len(scored)
    pct_ex  = 100.0 * n_exact / n_total if n_total else 0
    pct_cl  = 100.0 * (n_exact + n_close) / n_total if n_total else 0

    print()
    print(sep)
    print(f"  Beats scored : {n_total:>4}")
    print(f"  Exact match  : {n_exact:>4}  ({pct_ex:.1f}%)")
    print(f"  Close (±1st) : {n_close:>4}  (within ±1 semitone)")
    print(f"  Wrong        : {n_wrong:>4}")
    print(f"  Exact+Close  : {n_exact+n_close:>4}  ({pct_cl:.1f}%)")

    # Per-section breakdown
    sec_stats = {}
    for r in scored:
        s = r['section']
        if s not in sec_stats:
            sec_stats[s] = {'total': 0, 'exact': 0, 'close': 0}
        sec_stats[s]['total'] += 1
        if r['exact']:
            sec_stats[s]['exact'] += 1
        elif r['close']:
            sec_stats[s]['close'] += 1

    print()
    print(f"  {'Section':<24}  {'Beats':>5}  {'Exact':>6}  {'Close':>6}  {'Exact%':>7}")
    print('  ' + '─' * 54)
    for sec, st in sec_stats.items():
        pct = 100.0 * st['exact'] / st['total'] if st['total'] else 0
        print(f"  {sec:<24}  {st['total']:>5}  {st['exact']:>6}  {st['close']:>6}  {pct:>6.1f}%")
    print(sep)
    print()
    print("  Legend:  ✓ exact   ~ close (±1 semitone)   ✗ wrong   · rest")
    print()
    print("  Interpretation guide:")
    print("  • Exact > 70%  ->  JSON is good for the metered section")
    print("  • Exact < 40%  ->  likely a wrong --scale or --bpm")
    print("  • Intro always lower (free-tempo alaap has timing uncertainty)")
    print("  • Many '~' (close)  ->  komal/shuddha swara confusion; check scale")
    print(sep)

    # 7. Optional plot ────────────────────────────────────────────────────
    if hasattr(args, 'plot') and args.plot:
        _plot_verification(times_a, pitches, results, beat_sec,
                           base_sa, title, json_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert MP3 to Swar-Laya Studio JSON via pitch analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scale / Tonic (--scale):
  Tell the script which note is Sa (the tonic that the tanpura is tuned to).
  This is the most important argument: getting it right makes all note names correct.
  Note names: C  C#  D  D#  E  F  F#  G  G#  A  A#  B
  Optionally add octave:  G#4  C5  D3  (default octave = 4, range ~260-494 Hz)
  Equivalent Hz values:
    C=261.6  C#=277.2  D=293.7  D#=311.1  E=329.6  F=349.2
    F#=370.0  G=392.0  G#=415.3  A=440.0  A#=466.2  B=493.9

Taal names (--taal):
  Keherwa   8-beat  (most common for bhavgeet / light classical)
  Bhajani   8-beat  (devotional variant)
  Dadra     6-beat
  Rupak     7-beat  (starts on 3rd beat)
  Ektaal   12-beat
  Teentaal 16-beat  (main classical taal)

Instruments (--melody-instrument / --intro-instrument):
  harmonium  violin  flute  sitar  piano

Examples:
  python mp3_to_swarlaya.py "sakhi.mp3" --scale G# --taal Keherwa --bpm 75
  python mp3_to_swarlaya.py "raag.mp3"  --scale D  --taal Teentaal
  python mp3_to_swarlaya.py "bhajan.mp3" --scale A --melody-instrument harmonium --intro-instrument flute
  python mp3_to_swarlaya.py "ghazal.mp3" --scale C# --dedup --max-lines 16
        """
    )

    # Positional
    parser.add_argument('mp3_file', help='Path to the MP3 file')

    # Scale / pitch
    sg = parser.add_mutually_exclusive_group()
    sg.add_argument('--scale', metavar='NOTE',
                    help='Assumed Sa (tonic) note: C C# D D# E F F# G G# A A# B '
                         '(optionally with octave, e.g. G#4). '
                         'All pitches in the MP3 are interpreted relative to this tonic. '
                         'Use the note the tanpura is tuned to.')
    sg.add_argument('--sa',    metavar='HZ', type=float,
                    help='Assumed Sa (tonic) in Hz -- alternative to --scale '
                         '(e.g. --sa 415.3 is the same as --scale G#)')

    # Taal / tempo
    parser.add_argument('--taal',  metavar='NAME',
                        help='Taal name: Keherwa Bhajani Dadra Rupak Ektaal Teentaal')
    parser.add_argument('--beats', metavar='N', type=int,
                        choices=[4, 6, 7, 8, 10, 12, 16],
                        help='Beats per cycle override (used if --taal not given)')
    parser.add_argument('--bpm',   metavar='N', type=int,
                        help='Tempo in BPM (auto-detected if not specified)')

    # Instruments
    parser.add_argument('--melody-instrument', metavar='NAME',
                        default='harmonium', choices=VALID_INSTRUMENTS,
                        help='Instrument for Mukhda/Antara sections (default: harmonium)')
    parser.add_argument('--intro-instrument',  metavar='NAME',
                        default='violin',    choices=VALID_INSTRUMENTS,
                        help='Instrument for Intro/Alaap sections (default: violin)')

    # Output
    parser.add_argument('--title',     metavar='TEXT',
                        help='Song title (defaults to filename without extension)')
    parser.add_argument('--output',    metavar='PATH',
                        help='Output JSON path (defaults to <mp3_name>.json)')
    parser.add_argument('--max-lines', metavar='N', type=int, default=50,
                        help='Maximum lines in output JSON (default: 50)')
    parser.add_argument('--dedup',     action='store_true',
                        help='Merge consecutive identical cycles (reduces line count)')
    parser.add_argument('--intro-cycles', metavar='N', type=int,
                        help='Force exactly N cycles to be labelled as Intro '
                             '(overrides auto-detection). Use when the program '
                             'labels too many or too few cycles as Intro.')
    parser.add_argument('--intro-win-ms', metavar='MS', type=float, default=80.0,
                        help='Analysis window size for the free-tempo intro in ms '
                             '(default: 80). Shorter = captures faster ornaments.')
    parser.add_argument('--intro-hop-ms', metavar='MS', type=float, default=20.0,
                        help='Window hop for the free-tempo intro in ms '
                             '(default: 20). Smaller = finer resolution.')
    parser.add_argument('--intro-min-dur-ms', metavar='MS', type=float, default=150.0,
                        help='Minimum note sustain to survive the intro dedup pass '
                             '(default: 150 ms). Increase to suppress noise blips.')
    parser.add_argument('--intro-min-octave', metavar='N', type=int, default=-1,
                        help='Lowest octave allowed in intro output (default: -1). '
                             'Rejects tanpura/room-noise ghost notes at octave -2. '
                             'Use -2 to allow very-low mandra notes.')
    parser.add_argument('--intro-fmin', metavar='HZ', type=float,
                        help='Minimum pitch frequency for the intro-specific HPS re-analysis '
                             '(Hz). The intro region is re-analysed independently of the '
                             'main song so that a higher fmin can suppress the tanpura drone '
                             '(which sits at base_sa/2 and drowns out soft flute/sitar). '
                             'Default: base_sa * 0.55 (just above the tanpura fundamental). '
                             'Increase to e.g. 400 when the alaap starts on Pa (middle) or '
                             'higher and tanpura interference is severe. '
                             'Overridden by --intro-start if both are given.')
    parser.add_argument('--intro-start', metavar='NOTE',
                        help='Lowest swara expected in the intro/alaap (e.g. Pa, Ni, Sa). '
                             'Automatically computes the intro HPS fmin as one quarter-tone '
                             'below that swara\'s frequency, suppressing the tanpura drone '
                             'without requiring you to know the Hz value. '
                             'Valid values: ' + ' '.join(SWARA_NAMES) + '. '
                             'Overrides --intro-fmin when both are given. '
                             'Example: --intro-start Pa  (alaap opens on Pa, middle octave)')
    parser.add_argument('--song-start', metavar='NOTE',
                        help='Lowest swara expected in the metered song section (e.g. Ga, Ma). '
                             'Automatically computes the main HPS fmin as one quarter-tone '
                             'below that swara\'s frequency -- more musical than --fmin HZ. '
                             'Requires --scale or --sa to be effective (needs base_sa). '
                             'Valid values: ' + ' '.join(SWARA_NAMES) + '. '
                             'Overrides --fmin when both are given. '
                             'Example: --song-start Ga  (singer never goes below Ga)')
    parser.add_argument('--auto-meend', action='store_true',
                        help='Auto-detect meend (glide) on ascending stepwise intervals. '
                             'Off by default -- set meend manually in the studio after listening.')
    parser.add_argument('--fmin',      metavar='HZ', type=float,
                        help='Minimum pitch frequency for HPS detector (Hz). '
                             'Auto-computed from --scale when not given '
                             '(= base_sa * 0.52, which excludes the lower tanpura drone). '
                             'Override if the auto value cuts real melody notes.')
    parser.add_argument('--verbose',   action='store_true',
                        help='Print cycle-by-cycle note breakdown')

    # Verification mode
    parser.add_argument('--verify',    metavar='JSON_PATH',
                        help='Instead of generating a new JSON, compare an existing '
                             'JSON against the MP3 and report accuracy beat-by-beat. '
                             'Example:  --verify sakhi.json')
    parser.add_argument('--plot',      action='store_true',
                        help='With --verify: save a PNG chart showing MP3 pitch '
                             'contour vs JSON notes. Requires matplotlib.')

    args = parser.parse_args()

    # ── Verify mode: compare existing JSON vs MP3, then exit ──────────────
    if args.verify:
        run_verify(args)
        return

    # ------------------------------------------------------------------
    # Validate / derive inputs
    # ------------------------------------------------------------------
    if not os.path.isfile(args.mp3_file):
        sys.exit(f"Error: file not found: {args.mp3_file}")

    title    = args.title  or os.path.splitext(os.path.basename(args.mp3_file))[0]
    out_path = args.output or os.path.splitext(args.mp3_file)[0] + '.json'

    # ------------------------------------------------------------------
    # Early warnings for common mistakes with new note-name arguments
    # ------------------------------------------------------------------
    _intro_start = getattr(args, 'intro_start', None)
    _song_start  = getattr(args, 'song_start',  None)

    if _intro_start and not args.intro_cycles:
        print()
        print("  *** WARNING: --intro-start given but --intro-cycles is missing. ***")
        print(f"      --intro-start '{_intro_start}' only takes effect when --intro-cycles N")
        print("      is also specified.  Without --intro-cycles the free-tempo alaap")
        print("      analysis is skipped entirely and --intro-start is a no-op.")
        print()
        print("      Add --intro-cycles N where N = number of taal cycles in the alaap.")
        print("      Example: for a 13-second intro at 158 BPM Rupak (7 beats):")
        print("               13s / (7 * 60/158 s) = 5 cycles  ->  --intro-cycles 5")
        print()

    if _song_start and not (args.scale or args.sa):
        print()
        print("  *** WARNING: --song-start given but --scale / --sa is missing. ***")
        print(f"      --song-start '{_song_start}' needs base_sa (Hz) to compute fmin.")
        print("      Without --scale or --sa the argument falls back to 150 Hz default.")
        print("      Add --scale NOTE (e.g. --scale D#) for this to work correctly.")
        print()

    # Resolve taal
    taal_key, bpc = '8', 8     # defaults
    if args.taal:
        key = args.taal.lower().replace(' ', '')
        if key not in TAAL_MAP:
            avail = ', '.join(sorted({k.capitalize() for k in TAAL_MAP}))
            sys.exit(f"Error: unknown taal '{args.taal}'. Available: {avail}")
        taal_key, bpc = TAAL_MAP[key]
    if args.beats:             # --beats overrides taal's beat count
        bpc = args.beats
        if not args.taal:
            taal_key = str(bpc)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    sep = "=" * 62
    print(sep)
    print("  mp3_to_swarlaya  -  Swar-Laya Studio JSON generator")
    print(sep)
    print(f"  Input   : {args.mp3_file}")
    print(f"  Output  : {out_path}")
    print(f"  Title   : {title}")
    print(f"  Taal    : {args.taal or 'Keherwa (default)'}  "
          f"({bpc} beats, taal_key={taal_key!r})")
    print(f"  Melody  : {args.melody_instrument}")
    print(f"  Intro   : {args.intro_instrument}")
    print()

    # ------------------------------------------------------------------
    # [1] Load audio
    # ------------------------------------------------------------------
    print("[1/6] Loading MP3...")
    audio, sr = load_mp3(args.mp3_file)
    duration  = len(audio) / sr
    print(f"      Duration {duration:.1f}s   Sample rate {sr} Hz")

    # ------------------------------------------------------------------
    # [2/3] Sa resolution  (BEFORE pitch frames when --scale is given so
    #        we can set a smart fmin that excludes the tanpura drone)
    # ------------------------------------------------------------------
    print("[2/6] Assuming Sa (tonic)...")

    base_sa_early = None   # resolved here only when user gave --scale/--sa
    if args.scale:
        try:
            base_sa_early = parse_scale_arg(args.scale)
        except argparse.ArgumentTypeError as e:
            sys.exit(f"Error: {e}")
        print(f"      Assumed Sa = {args.scale}  ({base_sa_early:.2f} Hz)  [user-specified via --scale]")
    elif args.sa:
        base_sa_early = float(args.sa)
        print(f"      Assumed Sa = {base_sa_early:.2f} Hz  [user-specified via --sa]")
    else:
        print("      Sa not specified - will auto-detect after pitch analysis")
        print("      (Use --scale <note> for best accuracy, e.g. --scale G#)")

    # ------------------------------------------------------------------
    # Compute fmin for HPS.
    # Priority: --fmin HZ  >  --song-start NOTE  >  base_sa * 0.45  >  150 Hz
    #   base_sa * 0.45 admits melody in lower octave; tanpura rejected by
    #   the 5.5x confidence threshold.
    #   --song-start sets fmin just below the named swara so notes below the
    #   alaap/voice range are excluded at the detector level.
    # ------------------------------------------------------------------
    if args.fmin:
        pitch_fmin = args.fmin
        print(f"      fmin = {pitch_fmin:.1f} Hz  [user-specified via --fmin]")
    elif getattr(args, 'song_start', None) and base_sa_early is not None:
        _ss_note = args.song_start.strip()
        if _ss_note not in SWARA_IDX:
            sys.exit(f"Error: --song-start: unknown swara '{_ss_note}'. "
                     f"Valid values: {' '.join(SWARA_NAMES)}")
        _ss_hz     = base_sa_early * (2 ** (SWARA_IDX[_ss_note] / 12.0))
        pitch_fmin = _ss_hz * (2 ** (-0.5 / 12.0))   # one quarter-tone below
        print(f"      fmin = {pitch_fmin:.1f} Hz  "
              f"(just below {_ss_note} = {_ss_hz:.1f} Hz)  [from --song-start {_ss_note}]")
    elif getattr(args, 'song_start', None) and base_sa_early is None:
        pitch_fmin = 150.0
        print(f"      fmin = {pitch_fmin:.1f} Hz  (default - --song-start requires --scale or --sa)")
    elif base_sa_early is not None:
        pitch_fmin = base_sa_early * 0.45
        print(f"      fmin = {pitch_fmin:.1f} Hz  (base_sa x 0.45 - admits melody in lower octave; "
              f"tanpura rejected by confidence threshold 5.5x)")
    else:
        pitch_fmin = 150.0
        print(f"      fmin = {pitch_fmin:.1f} Hz  (default - re-run with --scale for cleaner results)")

    # ------------------------------------------------------------------
    # [3] Pitch detection  (now uses the smart fmin)
    # ------------------------------------------------------------------
    print("[3/6] Detecting pitches (HPS, 40 ms hop)...")
    times, pitches = compute_pitch_frames(audio, sr, hop_ms=40, frame_ms=60,
                                          fmin=pitch_fmin)
    valid_pct = 100 * sum(1 for p in pitches if p is not None) / max(len(pitches), 1)
    print(f"      {len(pitches)} frames, {valid_pct:.0f}% with confident pitch")

    # ------------------------------------------------------------------
    # Finalise base_sa  (auto-detect only if not given by user)
    # ------------------------------------------------------------------
    if base_sa_early is not None:
        base_sa = base_sa_early
    else:
        base_sa = detect_sa(pitches, verbose=args.verbose)
        print(f"      Auto-detected Sa = {base_sa:.2f} Hz  [use --scale to override]")

    scale_str   = sa_to_scale_str(base_sa)
    actual_sa   = float(scale_str) * 2
    note_label  = SCALE_TABLE.get(float(scale_str), '?')
    print(f"      All pitches mapped relative to Sa = {actual_sa:.2f} Hz ({note_label})")
    print(f"      JSON scale value = {scale_str!r}")

    if abs(base_sa - actual_sa) > 8:
        print(f"      [Note] Snapped {base_sa:.1f} Hz -> nearest chromatic {actual_sa:.1f} Hz")
        base_sa = actual_sa

    # ------------------------------------------------------------------
    # [4] BPM detection
    # ------------------------------------------------------------------
    print("[4/6] Resolving BPM...")

    if args.bpm:
        bpm = int(args.bpm)
        print(f"      BPM = {bpm}  (user-specified)")
    else:
        bpm = detect_bpm(audio, sr)
        print(f"      BPM = {bpm}  (auto-detected)")

    beat_sec    = 60.0 / bpm
    cycle_sec   = beat_sec * bpc
    n_cycles_ex = int(duration / cycle_sec)
    print(f"      Beat = {beat_sec:.3f}s   Cycle = {cycle_sec:.2f}s   "
          f"Expected cycles ~ {n_cycles_ex}")

    # ------------------------------------------------------------------
    # [5] Beat extraction + cycle grouping
    #
    #   When --intro-cycles N is given the intro region (free-tempo alaap)
    #   is extracted separately with a sliding-window + dedup approach that
    #   does NOT assume a beat grid.  The metered section starts at
    #   intro_end_time and is analysed with the normal beat-synchronous
    #   method -- but now the phase search is restricted to the metered
    #   region only, giving much cleaner alignment.
    # ------------------------------------------------------------------
    print("[5/6] Extracting beats and grouping into cycles...")

    times_a = np.array(times)

    if args.intro_cycles:
        intro_end_time = args.intro_cycles * bpc * beat_sec
        print(f"      Intro region : 0 - {intro_end_time:.1f}s "
              f"({args.intro_cycles} cycle(s), free-tempo analysis)")
        print(f"      Intro params : win={args.intro_win_ms:.0f}ms  "
              f"hop={args.intro_hop_ms:.0f}ms  "
              f"min_dur={args.intro_min_dur_ms:.0f}ms  "
              f"min_octave={args.intro_min_octave}")
        print(f"      Metered start: {intro_end_time:.1f}s "
              f"(beat-synchronous analysis)")

        # Re-analyse ONLY the intro slice with a higher fmin so that the
        # tanpura drone (at base_sa/2) does not drown out the soft
        # flute/sitar alaap.  The main `pitches` array (low fmin) is kept
        # for the beat-synchronous metered section.
        #
        # Priority: --intro-start NOTE  >  --intro-fmin HZ  >  base_sa * 0.55
        _intro_start = getattr(args, 'intro_start', None)
        if _intro_start:
            _is_note = _intro_start.strip()
            if _is_note not in SWARA_IDX:
                sys.exit(f"Error: --intro-start: unknown swara '{_is_note}'. "
                         f"Valid values: {' '.join(SWARA_NAMES)}")
            _is_hz        = base_sa * (2 ** (SWARA_IDX[_is_note] / 12.0))
            intro_fmin_hz = _is_hz * (2 ** (-0.5 / 12.0))   # one quarter-tone below
            _fmin_source  = f"just below {_is_note} = {_is_hz:.1f} Hz  [--intro-start {_is_note}]"
        elif args.intro_fmin:
            intro_fmin_hz = args.intro_fmin
            _fmin_source  = f"[--intro-fmin {intro_fmin_hz:.1f}]"
        else:
            intro_fmin_hz = base_sa * 0.55
            _fmin_source  = "default (base_sa x 0.55)"
        print(f"      Intro fmin   : {intro_fmin_hz:.1f} Hz  {_fmin_source}  "
              f"(tanpura drone at {base_sa/2:.1f} Hz suppressed)")
        end_sample = min(int((intro_end_time + 1.0) * sr), len(audio))
        i_times_raw, i_pitches_raw = compute_pitch_frames(
            audio[:end_sample], sr,
            hop_ms=20, frame_ms=40, fmin=intro_fmin_hz)
        i_times_a = np.array(i_times_raw)

        intro_notes  = extract_free_section(i_times_a, i_pitches_raw, base_sa,
                                            0.0, intro_end_time,
                                            win_sec    = args.intro_win_ms     / 1000.0,
                                            hop_sec    = args.intro_hop_ms     / 1000.0,
                                            min_dur_sec= args.intro_min_dur_ms / 1000.0,
                                            min_octave = args.intro_min_octave)
        intro_cycles_list = notes_to_cycles(intro_notes, bpc)
        print(f"      Intro: {len(intro_notes)} note events -> "
              f"{len(intro_cycles_list)} cycle(s)")

        beats  = extract_beats(times, pitches, base_sa, bpm, duration,
                               t_start=intro_end_time)
        metered_cycles = beats_to_cycles(beats, bpc)
        print(f"      Metered: {len(beats)} beats -> {len(metered_cycles)} cycle(s)")

        cycles = intro_cycles_list + metered_cycles
    else:
        beats  = extract_beats(times, pitches, base_sa, bpm, duration)
        cycles = beats_to_cycles(beats, bpc)
        print(f"      {len(beats)} beats   {len(cycles)} complete {bpc}-beat cycles")

    # ------------------------------------------------------------------
    # [6] Section labels + optional dedup + max-lines trim
    # ------------------------------------------------------------------
    print("[6/6] Labelling sections...")

    cycles = identify_sections(cycles, intro_cycles=args.intro_cycles)

    if args.dedup:
        before = len(cycles)
        cycles = deduplicate_consecutive(cycles)
        print(f"      --dedup: {before} cycles -> {len(cycles)} unique phrases")

    if len(cycles) > args.max_lines:
        print(f"      Capping at {args.max_lines} lines (--max-lines)")
        cycles = cycles[:args.max_lines]

    print(f"      Writing {len(cycles)} lines to JSON")

    # ------------------------------------------------------------------
    # Verbose cycle table
    # ------------------------------------------------------------------
    if args.verbose:
        print()
        hdr = f"  {'Cy':>3} | {'Time':>6} | {'Section':<22} | " + \
              '  '.join(f'B{i+1}' for i in range(bpc))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for i, cy in enumerate(cycles):
            t   = cy['start_time']
            lbl = cy.get('section', '')
            parts = []
            for note, oct_ in cy['data']:
                tag = '.' * min(-oct_, 2) if oct_ < 0 else '^' * min(oct_, 1)
                parts.append(f"{note}{tag}")
            row = '  '.join(f"{p:<5}" for p in parts)
            print(f"  {i+1:3d} | {t:5.1f}s | {lbl:<22} | {row}")
        print()

    # ------------------------------------------------------------------
    # Build and write JSON
    # ------------------------------------------------------------------
    song = build_json(cycles, title, scale_str, bpm,
                      taal_key, bpc,
                      args.melody_instrument, args.intro_instrument,
                      auto_meend=args.auto_meend)

    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(song, fout, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(sep)
    print("  RESULT")
    print(sep)
    print(f"  Written  : {out_path}")
    print(f"  Title    : {title}")
    print(f"  Scale    : {scale_str}  (Sa = {actual_sa:.2f} Hz, {note_label})")
    print(f"  BPM      : {bpm}")
    taal_display = args.taal or 'Keherwa'
    print(f"  Taal     : {taal_display} ({bpc} beats, taal_key={taal_key!r})")
    print(f"  Lines    : {len(song['lines'])}")
    print()
    print(f"  {'#':>3}  {'Section':<22}  {'Instrument':<12}  {'Tabla':<6}  Notes")
    print("  " + "-" * 74)
    for i, ln in enumerate(song['lines']):
        notes_s = ' '.join(n[0] for n in ln['notes'])
        tabla_s = 'off' if ln['tabla_mute'] else 'on '
        print(f"  {i+1:3d}  {ln['section']:<22}  {ln['line_instrument']:<12}  "
              f"{tabla_s}     {notes_s}")

    print()
    print("  TIP: Load this JSON in Swar-Laya Studio v3 with the Load button.")
    if not (args.scale or args.sa):
        print("  TIP: Re-run with --scale <note> if notes sound wrong.")
        print("       (Sa should match the tanpura drone note, e.g. --scale G#)")
    if not args.bpm:
        print(f"  TIP: Re-run with --bpm {bpm} to lock tempo if playback drifts.")
    print("  TIP: Meend (glide) is off by default. Add --auto-meend to enable")
    print("       auto-detection, or set it manually per note in the studio.")
    print(sep)


if __name__ == '__main__':
    main()
