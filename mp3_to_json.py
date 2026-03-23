"""
mp3_to_json.py  --  MP3 -> Swar-Laya Studio JSON
================================================

Converts any MP3 file to a complete Swar-Laya Studio JSON, ready to load
and play in the Studio.  Only high-confidence notes are filled in; all
other beats are left blank (-) for the user to correct by ear.

What it does automatically:
  -Separates vocals from tabla/instruments using Demucs (reuses existing
    stems if they are already in the --stems-dir folder)
  -Detects BPM via librosa beat-tracking
  -Detects the tanpura Sa (scale) via chromagram analysis of the vocal stem
  -Detects where the vocals begin by measuring vocal stem energy
  -Writes one JSON line per avartana covering the full song duration

Required argument:
  --taal INT    Beats per avartana  (e.g. 7 = Rupak, 8 = Keherwa, 16 = Teentaal)

All auto-detected values are printed so you can verify or override them.

Usage examples:
  # Let everything be auto-detected (only taal is required):
  python mp3_to_json.py "Sakhi-Mand-Jhalya-Taarka.mp3" --taal 7

  # Override BPM and scale if auto-detection is wrong:
  python mp3_to_json.py "Sakhi-Mand-Jhalya-Taarka.mp3" --taal 7 ^
      --bpm 152 --scale 155.56 --vocal-start 15.15 ^
      --title "Sakhi - Mand Jhalya Taarka"

Options:
  mp3               Path to the MP3 file  (required)
  --taal    INT     Beats per avartana  (required)
  --bpm     FLOAT   Override BPM (auto-detected if omitted)
  --scale   FLOAT   Override tanpura Sa in Hz (auto-detected if omitted)
  --vocal-start F   Override vocal start in seconds (auto-detected if omitted)
  --title   STR     Song title  (default: filename stem)
  --instrument STR  harmonium / flute / sitar / violin  (default: harmonium)
  --confidence F    Min voiced-confidence to emit a note, 0–1  (default: 0.80)
  -o / --output     Output JSON path  (default: <mp3_stem>.json next to the MP3)
  --stems-dir       Folder for Demucs stems  (default: 'separated/' next to MP3)
"""

import argparse, json, math, os, sys
import numpy as np
import librosa
import soundfile as sf

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Swara table ────────────────────────────────────────────────────────────────
SWARA_NAMES = ["Sa","re","Re","ga","Ga","Ma","ma","Pa","dha","Dha","ni","Ni"]

# Standard tanpura Sa values (Hz) mapped to Indian notation
KNOWN_SA = {
    130.81: "Pandhri 1 (C)",  138.59: "Kali 1 (C#)",
    146.83: "Pandhri 2 (D)",  155.56: "Kali 2 (D#/Eb)",
    164.81: "Pandhri 3 (E)",  174.61: "Pandhri 4 (F)",
    185.00: "Kali 3 (F#)",    196.00: "Pandhri 5 (G)",
    207.65: "Kali 4 (G#/Ab)", 220.00: "Pandhri 6 (A)",
    233.08: "Kali 5 (A#/Bb)", 246.94: "Pandhri 7 (B)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def freq_to_swara(f, playback_sa):
    if f <= 0 or math.isnan(f):
        return "-", 0
    raw = 12.0 * math.log2(f / playback_sa)
    si  = round(raw)
    return SWARA_NAMES[si % 12], si // 12


def nearest_sa(hz):
    """Round a detected Sa frequency to the nearest known tanpura value."""
    return min(KNOWN_SA, key=lambda x: abs(x - hz))


def blank_line(taal, instrument, percussion):
    return {
        "section": "", "line_instrument": instrument,
        "percussion": percussion, "line_volume": 1.0,
        "adlib": False, "line_bpm": None,
        "taal_key": str(taal), "beats": str(taal),
        "lyrics":  [""] * taal,
        "notes":   [["-"]] * taal,
        "octaves": [[0]]   * taal,
        "meend":   [[False]] * taal,
    }


def note_line(notes, octaves, taal, instrument, percussion):
    return {
        "section": "", "line_instrument": instrument,
        "percussion": percussion, "line_volume": 1.0,
        "adlib": False, "line_bpm": None,
        "taal_key": str(taal), "beats": str(taal),
        "lyrics":  [""] * taal,
        "notes":   [[n] for n in notes],
        "octaves": [[o] for o in octaves],
        "meend":   [[False]] * taal,
    }


# ── Demucs: separate vocals ────────────────────────────────────────────────────

def ensure_vocals_wav(mp3_path, stems_dir):
    """Return vocals.wav path, running Demucs if needed."""
    song  = os.path.splitext(os.path.basename(mp3_path))[0]
    dest  = os.path.join(stems_dir, "htdemucs", song)
    vpath = os.path.join(dest, "vocals.wav")

    if os.path.exists(vpath):
        print(f"  Vocal stem : {vpath}  (reusing)")
        return vpath

    print("  Running Demucs htdemucs (downloads ~80 MB model on first run)...")
    import torch
    from demucs.pretrained import get_model
    from demucs.apply      import apply_model

    model = get_model("htdemucs")
    model.eval()
    sr_m  = model.samplerate

    wav, _ = librosa.load(mp3_path, sr=sr_m, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])

    print(f"  Separating {wav.shape[1]/sr_m:.1f} s of audio...")
    with torch.no_grad():
        sources = apply_model(model, torch.tensor(wav[np.newaxis]).float(),
                              progress=True)

    os.makedirs(dest, exist_ok=True)
    for i, name in enumerate(model.sources):
        sf.write(os.path.join(dest, f"{name}.wav"),
                 sources[0, i].numpy().T, sr_m, subtype="PCM_16")
    print(f"  Stems saved: {dest}")
    return vpath


# ── Auto-detect BPM ───────────────────────────────────────────────────────────

def detect_bpm(mp3_path):
    print("  Detecting BPM from beat tracking...")
    y, sr = librosa.load(mp3_path, sr=22050, mono=True, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(round(float(tempo), 1))
    print(f"  Detected BPM: {bpm}")
    return bpm


# ── Auto-detect vocal start ───────────────────────────────────────────────────

def detect_vocal_start(vocals_wav, window_s=0.5, threshold_ratio=0.12):
    """Find first second where vocal stem RMS exceeds threshold_ratio × peak RMS."""
    print("  Detecting vocal start from vocal stem energy...")
    y, sr = librosa.load(vocals_wav, sr=22050, mono=True)
    hop   = int(window_s * sr)
    rms   = librosa.feature.rms(y=y, frame_length=hop*2, hop_length=hop)[0]
    peak  = rms.max()
    thresh = peak * threshold_ratio
    for i, r in enumerate(rms):
        if r > thresh:
            t = i * window_s
            print(f"  Detected vocal start: {t:.1f} s")
            return t
    print("  Vocal start not detected — using 0.0 s")
    return 0.0


# ── Auto-detect scale (Sa) ────────────────────────────────────────────────────

def detect_scale(vocals_wav, vocal_start_s, duration_s=60.0):
    """
    Estimate tanpura Sa from the first `duration_s` of the vocal section.
    Uses chromagram to find the dominant pitch class, then finds the most
    common frequency in that class among pYIN-voiced frames.
    """
    print(f"  Detecting Sa from vocal stem (first {duration_s:.0f} s of vocal section)...")
    SR = 22050
    y, sr = librosa.load(vocals_wav, sr=None, mono=True,
                         offset=vocal_start_s, duration=duration_s)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Chromagram → dominant pitch class
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR)
    dominant_class = int(np.argmax(chroma.mean(axis=1)))   # 0=C, 1=C#, …, 11=B

    # pYIN for absolute frequency
    f0, voiced, _ = librosa.pyin(y, fmin=100, fmax=700, sr=SR,
                                 frame_length=2048, hop_length=512)
    voiced_f0 = f0[voiced & ~np.isnan(f0)]

    if len(voiced_f0) < 10:
        print("  Not enough voiced frames — scale detection unreliable")
        return None

    # Filter to frequencies matching the dominant pitch class (± 50 cents)
    semitones  = 12 * np.log2(voiced_f0 / 261.63)   # relative to C4
    pitch_class = np.round(semitones).astype(int) % 12
    matching   = voiced_f0[pitch_class == dominant_class]

    if len(matching) < 5:
        print(f"  Few matching frames for class {dominant_class}")
        return None

    # The Sa should be in the range 110–280 Hz (tanpura fundamental)
    # Deduce octave: take the octave whose median is in [110, 280]
    for octave_shift in [-1, 0, 1, -2, 2]:
        shifted = matching / (2 ** octave_shift)
        candidates = shifted[(shifted >= 110) & (shifted <= 280)]
        if len(candidates) >= 3:
            detected_hz = float(np.median(candidates))
            snapped = nearest_sa(detected_hz)
            name    = KNOWN_SA[snapped]
            print(f"  Detected Sa: {detected_hz:.1f} Hz -> snapped to {snapped} Hz ({name})")
            return snapped

    print("  Scale detection uncertain — please provide --scale")
    return None


# ── Pitch detection on vocal stem ─────────────────────────────────────────────

def build_vocal_lines(vocals_wav, vocal_start_s, bpm, taal,
                      playback_sa, confidence, total_avars, n_intro):
    """
    Run pYIN on vocals.wav from vocal_start_s.
    Returns list of (notes, octaves) per avartana.
    """
    SR       = 22050
    HOP      = 512
    hop_s    = HOP / SR
    beat_dur = 60.0 / bpm
    avar_dur = taal * beat_dur
    n_voc    = total_avars - n_intro

    print(f"  Running pYIN on vocal stem  ({n_voc} avartanas)...")
    y, sr = librosa.load(vocals_wav, sr=None, mono=True)
    y     = y[int(vocal_start_s * sr):]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    fmin = max(100, playback_sa * 0.71)
    fmax = min(2000, playback_sa * 2.20)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=SR,
        frame_length=2048, hop_length=HOP,
    )

    lines = []
    for av in range(n_voc):
        notes, octs = [], []
        for b in range(taal):
            t0  = (av * taal + b) * beat_dur
            t1  = t0 + beat_dur
            fs  = int(t0 / hop_s)
            fe  = min(int(t1 / hop_s), len(f0))
            tot = fe - fs

            vf  = voiced_flag[fs:fe]
            pr  = voiced_probs[fs:fe]

            v_frac = float(np.sum(vf)) / tot  if tot > 0 else 0.0
            avg_p  = float(np.mean(pr))        if tot > 0 else 0.0

            # Both fraction-of-voiced-frames AND average probability must pass
            if v_frac >= confidence and avg_p >= confidence * 0.85:
                f0_win = f0[fs:fe][vf & ~np.isnan(f0[fs:fe])]
                if len(f0_win) > 0:
                    sw, oc = freq_to_swara(float(np.nanmedian(f0_win)), playback_sa)
                    notes.append(sw); octs.append(oc)
                    continue
            notes.append("-"); octs.append(0)

        lines.append((notes, octs))

    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        prog="mp3_to_json",
        description="MP3 → Swar-Laya Studio JSON via Demucs vocal isolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("mp3",              help="Source MP3 file")
    ap.add_argument("--taal",    "-t",  type=int,   required=True,
                    help="Beats per avartana (e.g. 7, 8, 16)")
    ap.add_argument("--bpm",     "-b",  type=float, default=None,
                    help="BPM (auto-detected if omitted)")
    ap.add_argument("--scale",   "-s",  type=float, default=None,
                    help="Tanpura Sa in Hz (auto-detected if omitted)")
    ap.add_argument("--vocal-start",    type=float, default=None,
                    help="Seconds where vocals begin (auto-detected if omitted)")
    ap.add_argument("--title",          default="",
                    help="Song title")
    ap.add_argument("--instrument",     default="harmonium",
                    choices=["harmonium","flute","sitar","violin","piano"])
    ap.add_argument("--confidence",     type=float, default=0.80,
                    help="Voiced confidence 0–1 (default: 0.80)")
    ap.add_argument("--output", "-o",   default=None,
                    help="Output JSON path")
    ap.add_argument("--stems-dir",      default=None,
                    help="Demucs stems directory (default: ./separated next to MP3)")
    args = ap.parse_args()

    mp3_abs   = os.path.abspath(args.mp3)
    if not os.path.exists(mp3_abs):
        sys.exit(f"ERROR: file not found: {mp3_abs}")

    mp3_dir   = os.path.dirname(mp3_abs)
    song      = os.path.splitext(os.path.basename(mp3_abs))[0]
    out_path  = args.output   or os.path.join(mp3_dir, f"{song}.json")
    stems_dir = args.stems_dir or os.path.join(mp3_dir, "separated")

    print("=" * 60)
    print(f"  MP3  : {mp3_abs}")
    print(f"  Taal : {args.taal} beats per avartana")
    print("=" * 60)

    # ── Step 1: Demucs ────────────────────────────────────────────────────────
    print("\n[1] Vocal stem separation")
    vocals_wav = ensure_vocals_wav(mp3_abs, stems_dir)

    # ── Step 2: Auto-detect or accept provided values ─────────────────────────
    print("\n[2] Song parameter detection")
    bpm         = args.bpm         or detect_bpm(mp3_abs)
    vocal_start = args.vocal_start if args.vocal_start is not None \
                                   else detect_vocal_start(vocals_wav)
    scale_hz    = args.scale       or detect_scale(vocals_wav, vocal_start)

    if scale_hz is None:
        print("\n  ERROR: Could not detect Sa frequency automatically.")
        print("  Please re-run with:  --scale <Hz>")
        print("  Common values: 130.81 (C) 138.59 (C#) 146.83 (D) 155.56 (D#)")
        print("                 164.81 (E) 174.61 (F)  185.00 (F#) 196.00 (G)")
        print("                 207.65 (G#) 220.00 (A) 233.08 (A#) 246.94 (B)")
        sys.exit(1)

    playback_sa = scale_hz * 2
    beat_dur    = 60.0 / bpm
    avar_dur    = args.taal * beat_dur

    print(f"\n  Final parameters:")
    print(f"    BPM          : {bpm}")
    print(f"    Scale (Sa)   : {scale_hz} Hz  [{KNOWN_SA.get(scale_hz,'custom')}]"
          f"  ->  playback Sa = {playback_sa:.2f} Hz")
    print(f"    Vocal start  : {vocal_start:.1f} s")
    print(f"    Taal         : {args.taal} beats  (avartana = {avar_dur:.3f} s)")
    print(f"    Confidence   : {args.confidence}")

    # ── Step 3: Song structure ────────────────────────────────────────────────
    total_dur  = librosa.get_duration(path=mp3_abs)
    n_total    = int(total_dur / avar_dur)          # full avartanas in the song
    n_intro    = int(vocal_start / avar_dur)        # avartanas before vocals
    n_vocal    = n_total - n_intro

    print(f"\n[3] Song structure")
    print(f"    Total duration  : {total_dur:.1f} s")
    print(f"    Total avartanas : {n_total}")
    print(f"    Intro avartanas : {n_intro}  (0 – {n_intro*avar_dur:.1f} s, all blank)")
    print(f"    Vocal avartanas : {n_vocal}  ({vocal_start:.1f} s – end)")

    # ── Step 4: Build intro lines (all blank) ─────────────────────────────────
    print("\n[4] Building intro lines (blank)")
    intro_lines = [blank_line(args.taal, args.instrument, "mute")
                   for _ in range(n_intro)]

    # ── Step 5: Pitch detect vocal section ───────────────────────────────────
    print("\n[5] Detecting pitches in vocal section")
    beat_data = build_vocal_lines(
        vocals_wav, vocal_start, bpm, args.taal,
        playback_sa, args.confidence, n_total, n_intro
    )

    vocal_lines = [
        note_line(ns, os_, args.taal, args.instrument, "tabla")
        for ns, os_ in beat_data
    ]

    all_lines = intro_lines + vocal_lines

    # ── Step 6: Write JSON ────────────────────────────────────────────────────
    out = {
        "title": args.title or song,
        "scale": str(scale_hz),
        "bpm":   int(round(bpm)),
        "lines": all_lines,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_voc_beats = n_vocal * args.taal
    note_beats  = sum(1 for l in vocal_lines for n in l["notes"] if n[0] != "-")
    blank_beats = total_voc_beats - note_beats

    print(f"\n{'='*60}")
    print(f"  Output : {out_path}")
    print(f"  Lines  : {len(all_lines)}  ({n_intro} blank intro + {n_vocal} vocal)")
    print(f"  Notes  : {note_beats}/{total_voc_beats} vocal beats filled"
          f"  ({100*note_beats/total_voc_beats:.0f}%)")
    print(f"  Blank  : {blank_beats} beats — fill by ear in the Studio")
    print(f"\n  Tip: open the isolated vocals file in a media player for easier")
    print(f"  manual transcription — tabla is completely removed from it:")
    voc_dir = os.path.join(stems_dir, "htdemucs", song)
    print(f"  {os.path.join(voc_dir, 'vocals.wav')}")
    print(f"{'='*60}")

    if args.bpm is None:
        print(f"\n  NOTE: BPM was auto-detected as {bpm}.")
        print(f"  If the beat grid looks wrong in the Studio, re-run with --bpm <value>.")
    if args.scale is None:
        print(f"\n  NOTE: Scale was auto-detected as {scale_hz} Hz.")
        print(f"  If notes sound wrong, re-run with --scale <Hz>.")
    if args.vocal_start is None:
        print(f"\n  NOTE: Vocal start was auto-detected at {vocal_start:.1f} s.")
        print(f"  If intro/blank lines seem off, re-run with --vocal-start <seconds>.")


if __name__ == "__main__":
    main()
