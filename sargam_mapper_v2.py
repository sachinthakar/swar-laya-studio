"""
sargam_mapper_v2 — vocal-first automatic notation for Swar-Laya Studio
=====================================================================

Why this replaces the pyin/YIN approach
----------------------------------------
The old detector tracked whatever periodic signal was loudest — usually the
harmonium/tanpura drone, or the sitar's own chikari strings in an intro — not
the singer. A cleaner MP3 does not fix that; the drone is a clean signal too.

This engine:
  1. (optional) separates the VOCAL stem with Demucs so pitch tracking sees the
     voice alone. Falls back to librosa harmonic separation if Demucs is absent.
  2. tracks pitch with CREPE (torchcrepe) — a neural tracker far more voice-
     accurate than pyin/YIN — with per-frame confidence.
  3. only trusts pitch where the voice is present; low-voice stretches (intro,
     interludes) are transcribed rough and FLAGGED "Instrumental — verify".
  4. places notes on the taal grid and, when a note is off the beat, splits the
     beat into 2/3/4 equal parts, e.g. ["-","Ga"] or ["Ni","Dha","-","-"].

Known limit: pure-instrumental drone passages defeat any monophonic tracker
(CREPE and Essentia both fail on the sitar intro). Those are flagged, not solved.

Use as a library:
    from sargam_mapper_v2 import transcribe
    doc = transcribe("song.mp3", sa="G#3", bpm=140, beats=6,
                     sam=2.38, vocal_start=10, demucs=True, model="full")

Or as a CLI:
    python sargam_mapper_v2.py "song.mp3" --sa G#3 --bpm 140 --beats 6 \
        --sam 2.38 --vocal-start 10 --demucs --wav
"""

import argparse, json, math, os
import numpy as np

# ── Swara tables (Just Intonation) ───────────────────────────────────────────
SWARA  = ["Sa","re","Re","ga","Ga","Ma","ma","Pa","dha","Dha","ni","Ni"]
JI     = [1/1,16/15,9/8,6/5,5/4,4/3,45/32,3/2,8/5,5/3,9/5,15/8]
JICENT = np.array([1200*math.log2(r) for r in JI])
NOTE_HZ = {"C":261.63,"C#":277.18,"D":293.66,"D#":311.13,"E":329.63,"F":349.23,
           "F#":369.99,"G":392.00,"G#":415.30,"A":440.00,"A#":466.16,"B":493.88}

def parse_sa(text):
    try:
        return float(text)
    except (ValueError, TypeError):
        pass
    t = str(text).strip().upper()
    octave = 4
    if t and t[-1].isdigit():
        octave = int(t[-1]); t = t[:-1]
    if t not in NOTE_HZ:
        raise ValueError(f"Unknown Sa note '{text}'. Use Hz or a name like G#3.")
    return NOTE_HZ[t] * (2 ** (octave - 4))

# ── Audio loading + vocal isolation ──────────────────────────────────────────
def load_audio(path, sr, max_seconds=None):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    if max_seconds:
        y = y[: int(max_seconds * sr)]
    return y

def isolate_vocal(path, sr, use_demucs, max_seconds=None, log=print, device="cpu"):
    import librosa
    if use_demucs:
        try:
            import torch, torchaudio
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            dev = resolve_device(device)
            log(f"  Separating vocal stem with Demucs on {dev} (first run downloads the model)…")
            model = get_model("htdemucs"); model.to(dev).eval()
            wav, in_sr = torchaudio.load(path)
            if max_seconds:
                wav = wav[:, : int(max_seconds * in_sr)]
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / (ref.std() + 1e-8)
            with torch.no_grad():
                sources = apply_model(model, wav[None], device=dev, progress=False)[0]
            voc = sources[model.sources.index("vocals")]
            voc = voc * (ref.std() + 1e-8) + ref.mean()
            voc = voc.mean(0).numpy()
            voc = librosa.resample(voc, orig_sr=in_sr, target_sr=sr)
            return voc.astype(np.float32), True
        except Exception as e:
            log(f"  Demucs unavailable ({e}); using harmonic separation instead.")
    y = load_audio(path, sr, max_seconds)
    return librosa.effects.harmonic(y.astype(np.float32), margin=3.0), False

# ── Pitch (CREPE) ────────────────────────────────────────────────────────────
def resolve_device(device):
    """'cpu', 'cuda', or 'auto' (use CUDA when available). GPU is future-ready:
    flip device to 'auto'/'cuda' once a CUDA torch build is installed."""
    if device in (None, "", "auto"):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device

def crepe_pitch(sig, sr, hop, fmin, fmax, model, device="cpu"):
    import torch, torchcrepe
    dev = resolve_device(device)
    f0, pdc = torchcrepe.predict(torch.tensor(sig)[None], sr, hop_length=hop,
                                 fmin=fmin, fmax=fmax, model=model, batch_size=512,
                                 device=dev, return_periodicity=True)
    return f0[0].numpy(), pdc[0].numpy()

def crepe_pitch_chunked(sig, sr, hop, fmin, fmax, model, device="cpu",
                        chunk_s=20.0, progress=None):
    """CREPE over a long signal in windows, so progress can be reported and
    memory stays bounded. Returns (f0, pd) on the full hop grid."""
    import numpy as np
    n = len(sig); win = int(chunk_s * sr)
    f0s, pds = [], []
    for off in range(0, n, win):
        chunk = sig[off:min(off + win, n)]
        if len(chunk) < hop * 2:
            break
        f0, pd = crepe_pitch(chunk, sr, hop, fmin, fmax, model, device)
        f0s.append(f0); pds.append(pd)
        if progress:
            progress(min(off + win, n) / n)
    if not f0s:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(f0s), np.concatenate(pds)

def crepe_voiced_only(sig, sr, hop, fmin, fmax, model, device="cpu",
                      progress=None, pad_s=0.3):
    """Speedup: run CREPE only over spans that carry vocal energy, leaving
    silent/instrumental gaps as unvoiced. Returns full-length (f0, pd)."""
    import numpy as np, librosa
    n_frames = 1 + (len(sig) - 1) // hop
    f0 = np.zeros(n_frames); pd = np.zeros(n_frames)
    rms = librosa.feature.rms(y=sig, frame_length=hop*4, hop_length=hop)[0]
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)))
    rms = rms[:n_frames]
    thr = max(rms.max() * 0.06, np.median(rms) * 0.5)
    voiced = rms > thr
    pad = int(pad_s * sr / hop)
    spans, i = [], 0
    while i < n_frames:
        if voiced[i]:
            j = i
            while j < n_frames and voiced[j]:
                j += 1
            a, b = max(0, i - pad), min(n_frames, j + pad)
            if spans and a <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], b))
            else:
                spans.append((a, b))
            i = j
        else:
            i += 1
    total = sum(b - a for a, b in spans) or 1
    done = 0
    for a, b in spans:
        clip = sig[a*hop: min(len(sig), b*hop + hop)]
        if len(clip) < hop * 2:
            continue
        cf, cp = crepe_pitch(clip, sr, hop, fmin, fmax, model, device)
        L = min(len(cf), b - a)
        f0[a:a+L] = cf[:L]; pd[a:a+L] = cp[:L]
        done += b - a
        if progress:
            progress(done / total)
    return f0, pd

# ── Sa / Sam detection ───────────────────────────────────────────────────────
def detect_sa(f0, voiced):
    from scipy.ndimage import gaussian_filter1d
    fv = f0[voiced & (f0 > 0)]
    if len(fv) < 200:
        return None
    cents = 1200 * np.log2(fv / 55.0)
    h = np.zeros(1200)
    for c in np.mod(cents, 1200):
        h[int(c) % 1200] += 1
    hs = gaussian_filter1d(np.concatenate([h, h, h]), 8)[1200:2400]
    sa = 55 * 2 ** (int(np.argmax(hs)) / 1200)
    while sa < 150:
        sa *= 2
    return sa

def detect_sam(path, sr, bpm, max_seconds=None):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    if max_seconds:
        y = y[: int(max_seconds * sr)]
    yp = librosa.effects.percussive(y, margin=3.0)
    hop = 256
    oenv = librosa.onset.onset_strength(y=yp, sr=sr, hop_length=hop)
    ot = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=hop)
    beat = 60.0 / bpm; dur = len(y) / sr
    best = None
    for sam in np.arange(0.0, beat, 0.02):
        k, s = 0, 0.0
        while sam + k * beat < dur:
            i = int(np.argmin(np.abs(ot - (sam + k * beat))))
            s += oenv[i]; k += 1
        s /= max(1, k)
        if best is None or s > best[0]:
            best = (s, sam)
    return float(best[1])

# ── Per-frame swara + sub-beat placement ─────────────────────────────────────
def frame_swaras(f0, pdc, sa_hz, conf=0.5, smooth=3):
    voiced = pdc > conf
    cents = np.where(voiced & (f0 > 0), 1200 * np.log2(np.maximum(f0, 1e-6) / sa_hz), np.nan)
    cs = cents.copy()
    for i in range(len(cs)):
        lo, hi = max(0, i - smooth), min(len(cs), i + smooth + 1)
        seg = cents[lo:hi]; seg = seg[~np.isnan(seg)]
        cs[i] = np.median(seg) if len(seg) else np.nan
    idx = np.full(len(cs), None, dtype=object)
    for i in range(len(cs)):
        if np.isnan(cs[i]):
            continue
        oc = int(math.floor(cs[i] / 1200)); base = cs[i] - oc * 1200
        j = int(np.argmin(np.abs(JICENT - base)))
        if base > 1150 and (1200 - base) < abs(base - JICENT[j]):
            j = 0; oc += 1
        idx[i] = oc * 12 + j
    return idx

def beat_subnotes(idx, fps, t0, beat, slots=12, tol=0.82):
    s = []
    for j in range(slots):
        fi = int(round((t0 + (j + 0.5) / slots * beat) * fps))
        s.append(idx[fi] if 0 <= fi < len(idx) else None)
    def build(D):
        per = slots // D; out = []
        for p in range(D):
            seg = [x for x in s[p*per:(p+1)*per] if x is not None]
            out.append(max(set(seg), key=seg.count) if seg else None)
        return out
    def agreement(D, sub):
        per = slots // D; seq = []
        for x in sub:
            seq += [x] * per
        return sum(1 for a, b in zip(seq, s) if a == b) / slots
    best = None
    for D in (1, 2, 3, 4):
        sub = build(D); sc = agreement(D, sub)
        if sc >= tol:
            return sub
        if best is None or sc > best[0]:
            best = (sc, sub)
    return best[1]

def make_line(cycle_subnotes, section, taal_key, instrumental, carry):
    notes, octs, lyr, meend = [], [], [], []
    for sub in cycle_subnotes:
        bn, bo = [], []
        for x in sub:
            if x is None:
                bn.append("-"); bo.append(0)
            elif carry[0] is not None and x == carry[0]:
                bn.append("~"); bo.append(0)
            else:
                bn.append(SWARA[x % 12]); bo.append(int(x // 12)); carry[0] = x
        if len(bn) > 1 and all(t == "~" for t in bn):
            bn, bo = ["~"], [0]
        if len(bn) > 1 and all(t == "-" for t in bn):
            bn, bo = ["-"], [0]; carry[0] = None
        notes.append(bn); octs.append(bo); lyr.append(""); meend.append([False]*len(bn))
    return {"section": section, "line_instrument": "harmonium",
            "percussion": "mute" if instrumental else "tabla", "line_volume": 1.0,
            "adlib": instrumental, "legato": instrumental, "line_bpm": None,
            "taal_key": taal_key, "beats": str(len(notes)), "lyrics": lyr,
            "notes": notes, "octaves": octs, "meend": meend}

def beat_cell(sub):
    """One beat's sub-note list -> (notes, octs) WITHOUT sustain (~) collapse.
    The frontend applies sustain across beats via its existing smApplySustain."""
    names, octs = [], []
    for x in sub:
        if x is None:
            names.append("-"); octs.append(0)
        else:
            names.append(SWARA[x % 12]); octs.append(int(x // 12))
    if len(names) > 1 and len(set(zip(names, octs))) == 1:      # all identical -> single
        names, octs = [names[0]], [octs[0]]
    if len(names) > 1 and all(n == "-" for n in names):        # all rest -> single rest
        names, octs = ["-"], [0]
    return names, octs

def beats_for_window(f0, pd, sr, hop, sa_hz, start_sec, n_beats, bpm):
    """Per-beat aggregates for one segment, from a pitch track (f0,pd) that is
    aligned so that index 0 corresponds to time `start_sec` of the song.
    Returns a list of {notes, octs, voiced_pct} — the shape smAggregateBeat
    produces, so the Studio's display/accept/edit flow is unchanged."""
    fps = sr / hop
    beat = 60.0 / bpm
    idx = frame_swaras(f0, pd, sa_hz)
    out = []
    for i in range(n_beats):
        t0 = i * beat
        f_lo = int(round(t0 * fps)); f_hi = int(round((t0 + beat) * fps))
        voiced_pct = int(100 * np.mean(pd[f_lo:min(f_hi, len(pd))] > 0.5)) if f_hi > f_lo else 0
        sub = beat_subnotes(idx, fps, t0, beat)
        n, o = beat_cell(sub)
        out.append({"notes": n, "octs": o, "voiced_pct": voiced_pct})
    return out

def detect_segment(vocal, sr, sa_hz, start_sec, n_beats, bpm, model="full",
                   device="cpu", hop=160, f0_cache=None, pd_cache=None):
    """Line-by-line detection for ONE segment. If a whole-file pitch cache is
    supplied (f0_cache/pd_cache on the `hop` grid from t=0), slices it (instant);
    otherwise runs CREPE only on this short slice (fast). Returns beat dicts."""
    beat = 60.0 / bpm
    fmin = max(70.0, sa_hz * 0.4); fmax = sa_hz * 3.2
    if f0_cache is not None and pd_cache is not None:
        fps = sr / hop
        f_lo = max(0, int(round(start_sec * fps)))
        f_hi = min(len(f0_cache), int(round((start_sec + n_beats * beat) * fps)) + 1)
        return beats_for_window(f0_cache[f_lo:f_hi], pd_cache[f_lo:f_hi], sr, hop,
                                sa_hz, 0.0, n_beats, bpm)
    s0 = max(0, int(start_sec * sr))
    s1 = min(len(vocal), int((start_sec + n_beats * beat) * sr) + hop)
    clip = vocal[s0:s1]
    if len(clip) < hop * 2:
        return [{"notes": ["-"], "octs": [0], "voiced_pct": 0} for _ in range(n_beats)]
    f0, pd = crepe_pitch(clip, sr, hop, fmin, fmax, model, device)
    return beats_for_window(f0, pd, sr, hop, sa_hz, 0.0, n_beats, bpm)

def vocal_energy_gate(sig, hop, n_frames):
    import librosa
    rms = librosa.feature.rms(y=sig, frame_length=hop*4, hop_length=hop)[0]
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)), constant_values=0)
    rms = rms[:n_frames]
    thr = max(rms.max() * 0.08, np.median(rms) * 0.5)
    return rms > thr

# ── Top-level transcription ──────────────────────────────────────────────────
def transcribe(audio_path, sa="auto", bpm=None, beats=6, sam="auto",
               vocal_start="auto", demucs=False, model="full",
               max_seconds=None, title=None, log=print, device="cpu",
               vocal_only=True, progress=None):
    """Run the full-song pipeline and return the Swar-Laya Studio JSON dict.
    (For the line-by-line button use detect_segment instead — far cheaper.)"""
    if bpm is None:
        raise ValueError("bpm is required")
    bpm = float(bpm)
    SR, HOP = 16000, 160
    fps = SR / HOP
    beat = 60.0 / bpm
    taal_key = str(int(beats))

    log("Loading + isolating vocal…")
    sig, used_demucs = isolate_vocal(audio_path, SR, demucs, max_seconds, log=log, device=device)

    sa_prov = 220.0 if sa == "auto" else parse_sa(sa)
    log(f"Tracking pitch with CREPE ({model}, {resolve_device(device)}, "
        f"{'vocal-only' if vocal_only else 'full'})…")
    fmin, fmax = max(70, sa_prov*0.4), sa_prov*3.2
    if vocal_only:
        f0, pdc = crepe_voiced_only(sig, SR, HOP, fmin, fmax, model, device, progress)
    else:
        f0, pdc = crepe_pitch_chunked(sig, SR, HOP, fmin, fmax, model, device, progress=progress)
    voiced = pdc > 0.5

    sa_hz = detect_sa(f0, voiced) if sa == "auto" else parse_sa(sa)
    if sa_hz is None:
        sa_hz = sa_prov
    log(f"  Sa = {sa_hz:.1f} Hz")

    sam_t = detect_sam(audio_path, 44100, bpm, max_seconds) if sam == "auto" else float(sam)
    log(f"  Sam = {sam_t:.2f} s   beat = {beat:.3f} s   {beats} beats/cycle")

    vgate = vocal_energy_gate(sig, HOP, len(f0))
    if vocal_start == "auto":
        on = np.where(vgate)[0]
        v_start = on[0] / fps if len(on) else 0.0
    else:
        v_start = float(vocal_start)
    log(f"  Vocal starts ~{v_start:.1f} s")

    idx = frame_swaras(f0, pdc, sa_hz)
    dur = len(f0) / fps
    lines, carry = [], [None]

    if sam_t > beat * 0.5:
        n = max(1, int(round(sam_t / beat)))
        cyc = [beat_subnotes(idx, fps, i*beat, beat) for i in range(n)]
        lines.append(make_line(cyc, "Intro / instrumental — verify", taal_key, True, carry))

    t = sam_t
    while t < dur - beat * 0.5:
        cyc = [beat_subnotes(idx, fps, t + b*beat, beat) for b in range(beats)]
        f_lo, f_hi = int(t*fps), int((t + beats*beat)*fps)
        voc_frac = float(vgate[f_lo:min(f_hi, len(vgate))].mean()) if f_hi > f_lo else 0.0
        instrumental = bool((t < v_start - 0.3) or (voc_frac < 0.35))
        sec = "Instrumental — verify" if instrumental else ""
        lines.append(make_line(cyc, sec, taal_key, instrumental, carry))
        t += beats * beat

    return {"title": title or os.path.splitext(os.path.basename(audio_path))[0],
            "scale": str(round(sa_hz, 2)), "bpm": int(round(bpm)), "pre_roll": 0,
            "lines": lines,
            "_meta": {"sa_hz": round(sa_hz, 2), "sam": round(sam_t, 3),
                      "vocal_start": round(v_start, 2), "demucs": used_demucs,
                      "model": model, "beats_per_cycle": beats}}

def write_resynth(doc, sam, bpm, sa_hz, path):
    import soundfile as sf
    SR = 22050; beat = 60.0/bpm
    nidx = {n: i for i, n in enumerate(SWARA)}
    total = sam + sum(len(ln["notes"]) * beat for ln in doc["lines"])
    audio = np.zeros(int((total + 1) * SR)); prev = None; t = sam
    for ln in doc["lines"]:
        for bn, bo in zip(ln["notes"], ln["octaves"]):
            L = len(bn); sub = beat / L
            for j, (nm, oc) in enumerate(zip(bn, bo)):
                st = t + j*sub
                if nm == "-":
                    prev = None; continue
                freq = prev if nm == "~" else sa_hz * JI[nidx[nm]] * (2**oc)
                if freq is None:
                    continue
                if nm != "~":
                    prev = freq
                n = int(sub*SR); tt = np.arange(n)/SR
                tone = 0.5*np.sin(2*np.pi*freq*tt) + 0.15*np.sin(2*np.pi*2*freq*tt)
                env = np.ones(n); a = min(int(0.008*SR), n//4); d = min(int(0.02*SR), n//4)
                if nm != "~" and a > 0:
                    env[:a] = np.linspace(0, 1, a)
                if d > 0:
                    env[-d:] = np.linspace(1, 0, d)
                s0 = int(st*SR); k = min(n, len(audio)-s0)
                if s0 >= 0:
                    audio[s0:s0+k] += (tone*env)[:k]*0.55
            t += beat
    sf.write(path, audio, SR)

def main():
    ap = argparse.ArgumentParser(description="Vocal-first Indian notation (Swar-Laya)")
    ap.add_argument("audio")
    ap.add_argument("--sa", default="auto")
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--beats", type=int, default=6)
    ap.add_argument("--sam", default="auto")
    ap.add_argument("--vocal-start", default="auto")
    ap.add_argument("--demucs", action="store_true")
    ap.add_argument("--model", default="full", choices=["full", "tiny"])
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto (GPU when available)")
    ap.add_argument("--whole-song", action="store_true",
                    help="process every frame (default: vocal-only spans, faster)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--wav", action="store_true")
    a = ap.parse_args()
    doc = transcribe(a.audio, sa=a.sa, bpm=a.bpm, beats=a.beats, sam=a.sam,
                     vocal_start=a.vocal_start, demucs=a.demucs, model=a.model,
                     max_seconds=a.max_seconds, device=a.device,
                     vocal_only=not a.whole_song,
                     progress=lambda p: print(f"\r  pitch {int(p*100):3d}%", end="", flush=True))
    print()
    out = os.path.splitext(a.audio)[0] + "_swarlekh.json"
    json.dump(doc, open(out, "w"), indent=1)
    print(f"Wrote {out}  ({len(doc['lines'])} lines)")
    if a.wav:
        m = doc["_meta"]
        write_resynth(doc, m["sam"], a.bpm, m["sa_hz"],
                      os.path.splitext(a.audio)[0] + "_resynth.wav")
        print("Wrote resynth wav")

if __name__ == "__main__":
    main()
