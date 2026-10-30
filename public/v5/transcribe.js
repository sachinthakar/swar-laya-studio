// Swar-Laya Studio v5 — transcribe from audio into the song sheet

// ═══════════════════════════════════════════════════════════════════════════════
//  TRANSCRIBE FROM AUDIO  (replaces the former "Sargam Mapper" tab)
//
//  Load a recording → set Start / Ad-lib / Lines → Detect. Detected notes are
//  written straight into the Composer song sheet as DRAFT lines (dashed amber).
//  Edit them with the normal Composer controls, then Accept (or Discard / Skip).
//  Sa, Taal (beats per line) and BPM are read from the Composer toolbar, so
//  there is one set of song settings, one editor, one playback engine and one
//  save/export path. Each transcribed line keeps its position in the recording
//  (audio_start / audio_end) for ▶ Orig and ⟲ Re-detect.
// ═══════════════════════════════════════════════════════════════════════════════

const SM_SWARA_NAMES  = ['Sa','re','Re','ga','Ga','Ma','ma','Pa','dha','Dha','ni','Ni'];
const SM_SECTION_GAP  = 0.1;   // seconds between the end of one section and the next Start

const SM = {
    audioCtx: null, audioBuffer: null, audioFile: null,
    rawPitchTrack: null,   // Float32Array, raw YIN Hz per frame (no gate, no smoothing)
    energyTrack:   null,   // Float32Array, RMS energy per frame
    pitchTrack:    null,   // gated + smoothed Hz (rebuilt on each Detect)
    hopLen: 512, frameLen: 2048, sampleRate: 44100,
    // Current section
    segStart: 0, segEnd: 0, segBeats: 0, segBpl: 0, segAdlibSec: 0, segAdlibBeats: 0,
    // Sustain carry-over from the previously accepted section
    carryNote: null, carryOct: 0,
    // Waveform interaction / original-audio playback
    dragging: false, play: null, playheadT: -1, animFrameId: null,
    // Drafts: 'section' (from Detect) or 'redetect' (one line re-detected in place)
    draftMode: null, replacedLine: null,
    // AI backend session
    sessionId: null, sessionReady: false,
};

// ── YIN pitch detection ────────────────────────────────────────────────────────
function yinDifference(buf, W) {
    const d = new Float32Array(W);
    for (let tau = 1; tau < W; tau++) {
        let s = 0;
        for (let j = 0; j < W; j++) {
            const diff = buf[j] - buf[j + tau];
            s += diff * diff;
        }
        d[tau] = s;
    }
    return d;
}

function yinCMND(d) {
    const cmnd = new Float32Array(d.length);
    cmnd[0] = 1;
    let runSum = 0;
    for (let tau = 1; tau < d.length; tau++) {
        runSum += d[tau];
        cmnd[tau] = runSum === 0 ? 0 : d[tau] * tau / runSum;
    }
    return cmnd;
}

function yinDetectFrame(frame, sampleRate, threshold = 0.20) {
    const W     = Math.floor(frame.length / 2);
    const d     = yinDifference(frame, W);
    const cmnd  = yinCMND(d);
    // Search within melodic range 80–1200 Hz (covers mandra to taar+ octaves)
    const tauMin = Math.max(2, Math.ceil(sampleRate / 1200)); // ≈19 at 22050 Hz → 1200 Hz upper
    const tauMax = Math.min(W - 1, Math.floor(sampleRate / 80)); // ≈275 at 22050 Hz → 80 Hz lower
    // Find first dip below threshold within the musical pitch range
    for (let tau = tauMin; tau < tauMax; tau++) {
        if (cmnd[tau] < threshold) {
            // Walk to the local minimum of this dip
            while (tau + 1 < tauMax && cmnd[tau + 1] < cmnd[tau]) tau++;
            // Parabolic interpolation for sub-sample accuracy
            const prev  = cmnd[tau - 1];
            const curr  = cmnd[tau];
            const next  = tau + 1 < W ? cmnd[tau + 1] : curr;
            const denom = 2 * (2 * curr - prev - next);
            const betterTau = (!denom || !isFinite(denom)) ? tau : tau + (prev - next) / denom;
            const hz = sampleRate / Math.max(1, betterTau);
            return (hz >= 80 && hz <= 1200) ? hz : 0;
        }
    }
    return 0; // unvoiced
}

// Compute full pitch track for audio samples (Float32Array, mono, normalized)
function smComputePitchTrack(samples, sampleRate) {
    const hop = SM.hopLen;
    const frame = SM.frameLen;
    const nFrames = Math.floor((samples.length - frame) / hop) + 1;
    const track = new Float32Array(nFrames);
    // Hann window
    const hann = new Float32Array(frame);
    for (let i = 0; i < frame; i++) hann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (frame - 1)));

    for (let fi = 0; fi < nFrames; fi++) {
        const start = fi * hop;
        const buf = new Float32Array(frame);
        for (let i = 0; i < frame; i++) buf[i] = samples[start + i] * hann[i];
        const hz = yinDetectFrame(buf, sampleRate);
        // Filter to vocal/melodic range 80–700 Hz
        track[fi] = (hz >= 80 && hz <= 700) ? hz : 0;
    }
    return track;
}

// Smooth pitch track with median-like window (suppress outliers)
function smSmoothPitch(track, w = 5) {
    const out = new Float32Array(track.length);
    const half = Math.floor(w / 2);
    for (let i = 0; i < track.length; i++) {
        const slice = [];
        for (let j = i - half; j <= i + half; j++) {
            if (j >= 0 && j < track.length && track[j] > 0) slice.push(track[j]);
        }
        if (slice.length === 0) { out[i] = 0; continue; }
        slice.sort((a, b) => a - b);
        out[i] = slice[Math.floor(slice.length / 2)];
    }
    return out;
}

// ── Beat aggregation ───────────────────────────────────────────────────────────
function smSemitone(hz, saHz) {
    if (hz <= 0 || saHz <= 0) return null;
    return Math.round(12 * Math.log2(hz / saHz));
}

function smAggregateBeat(pitchSlice, saHz, minSubFrac = 0.25) {
    const voiced = pitchSlice.filter(h => h > 0);
    if (voiced.length === 0) return { notes: ['-'], octs: [0], dom_hz: 0, voiced_pct: 0, clusters: [] };

    const voicedPct = Math.round(100 * voiced.length / pitchSlice.length);

    // Convert to semitones relative to madhya Sa
    const semitones = voiced.map(h => smSemitone(h, saHz)).filter(s => s !== null);
    if (semitones.length === 0) return { notes: ['-'], octs: [0], dom_hz: 0, voiced_pct: voicedPct, clusters: [] };

    // Dominant by majority vote
    const counts = {};
    semitones.forEach(s => counts[s] = (counts[s] || 0) + 1);
    let dominant = parseInt(Object.entries(counts).sort((a,b) => b[1]-a[1])[0][0]);

    // Vibrato absorption: snap ±1 semitone to dominant
    const clean = semitones.map(s => Math.abs(s - dominant) <= 1 ? dominant : s);

    // Clusters (top 3 for diagnostics)
    const cleanCounts = {};
    clean.forEach(s => cleanCounts[s] = (cleanCounts[s] || 0) + 1);
    const total = clean.length;
    const clusters = Object.entries(cleanCounts)
        .sort((a,b) => b[1]-a[1]).slice(0,3)
        .map(([sem, cnt]) => {
            const s = parseInt(sem);
            const hz = saHz * Math.pow(2, s / 12);
            return {
                hz: hz, pct: Math.round(100*cnt/total),
                note: SM_SWARA_NAMES[((s % 12) + 12) % 12],
                oct:  Math.floor(s / 12)
            };
        });

    // RLE on cleaned semitones
    const minFrames = Math.max(1, Math.round(minSubFrac * total));
    const rle = [];
    let curr = clean[0], cnt = 1;
    for (let i = 1; i < clean.length; i++) {
        if (clean[i] === curr) cnt++;
        else { rle.push([curr, cnt]); curr = clean[i]; cnt = 1; }
    }
    rle.push([curr, cnt]);

    const sig = rle.filter(([,c]) => c >= minFrames);
    const final = sig.length ? sig : [[dominant, 1]];

    const notes = final.map(([s]) => SM_SWARA_NAMES[((s % 12) + 12) % 12]);
    const octs  = final.map(([s]) => Math.floor(s / 12));
    const domHz = saHz * Math.pow(2, dominant / 12);

    return { notes, octs, dom_hz: domHz, voiced_pct: voicedPct, clusters };
}
// Apply sustain carry-over: replace '-' when prior note should sustain
function smApplySustain(beats, carryNote, carryOct) {
    const out = beats.map(b => Object.assign({}, b, {notes: [...b.notes], octs: [...b.octs]}));
    let lastN = carryNote, lastO = carryOct;
    for (let i = 0; i < out.length; i++) {
        const n = out[i].notes[0];
        if (n === '-' && lastN && lastN !== '-') {
            out[i].notes = ['~'];
            out[i].octs  = [lastO];
        } else if (n !== '-' && n !== '~') {
            lastN = out[i].notes[out[i].notes.length - 1];
            lastO = out[i].octs[out[i].octs.length - 1];
        }
    }
    return out;
}

// ── Settings shared with the Composer toolbar ──────────────────────────────────
function smBpm()          { return parseInt(document.getElementById('bpm').value) || 90; }
function smTaalKey()      { return document.getElementById('taal-select').value || ''; }
function smBeatsPerLine() { const k = smTaalKey(); return k ? (TaalBeats[k] || parseInt(k) || 0) : 0; }
function smScaleSa()      { return parseFloat(document.getElementById('scale-select').value) || 261.63; }
// Sa used for pitch analysis = toolbar Sa shifted by "Melody octave".
// Detected octaves are relative to this Sa; the Composer plays them at the toolbar Sa.
function smDetectSa() {
    return smScaleSa() * Math.pow(2, parseInt(document.getElementById('sm-oct-offset').value) || 0);
}
function smSetStatus(id, msg) { const el = document.getElementById(id); if (el) el.textContent = msg; }

// ── Pitch → beats for one stretch of audio ─────────────────────────────────────
function smProcessSegment(startSec, nBeats, beatDur) {
    const secPerFrame = SM.hopLen / SM.sampleRate;
    const saHz  = smDetectSa();
    const track = SM.pitchTrack;
    const beats = [];
    for (let i = 0; i < nBeats; i++) {
        const bs = startSec + i * beatDur, be = bs + beatDur;
        const fs = Math.max(0, Math.floor(bs / secPerFrame));
        const fe = Math.min(track.length, Math.floor(be / secPerFrame));
        beats.push(fe <= fs
            ? { notes: ['-'], octs: [0], dom_hz: 0, voiced_pct: 0, clusters: [] }
            : smAggregateBeat(Array.from(track.slice(fs, fe)), saHz));
    }
    return beats;
}

// ── Panel toggle ───────────────────────────────────────────────────────────────
function smToggleTranscribe(on) {
    document.getElementById('sm-panel').classList.toggle('open', on);
    document.getElementById('transcribe-label').classList.toggle('on', on);
    document.getElementById('transcribe-toggle').checked = on;
    try { localStorage.setItem('sl_transcribe', on ? '1' : '0'); } catch (e) {}
    if (on) { smSyncSectionEnd(); setTimeout(smDrawWaveform, 50); }
}

// ── Audio loading ──────────────────────────────────────────────────────────────
async function smLoadFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    SM.audioFile = file; SM.sessionId = null; SM.sessionReady = false;
    if (SMB.enabled()) SMB.openSession(file);
    smSetStatus('sm-load-status', '⏳ Loading audio…');
    if (!SM.audioCtx) SM.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    SM.audioBuffer = await SM.audioCtx.decodeAudioData(await file.arrayBuffer());
    SM.sampleRate  = SM.audioBuffer.sampleRate;
    smSetStatus('sm-load-status', `⏳ Computing pitch track… (${SM.audioBuffer.duration.toFixed(1)}s audio)`);

    // Mono mix (average channels) → downsample with box-FIR anti-alias filter
    const ch0 = SM.audioBuffer.getChannelData(0);
    const ch1 = SM.audioBuffer.numberOfChannels > 1 ? SM.audioBuffer.getChannelData(1) : null;
    let mono = ch0;
    if (ch1) { mono = new Float32Array(ch0.length); for (let i = 0; i < ch0.length; i++) mono[i] = (ch0[i] + ch1[i]) * 0.5; }
    let samples = mono, sr = SM.sampleRate;
    if (sr > 16000) {
        const factor = Math.max(2, Math.round(sr / 11025));
        const ds = new Float32Array(Math.floor(mono.length / factor));
        for (let i = 0; i < ds.length; i++) {
            let sum = 0;
            for (let k = 0; k < factor; k++) sum += mono[i * factor + k] || 0;
            ds[i] = sum / factor;
        }
        samples = ds; sr = Math.round(sr / factor); SM.sampleRate = sr;
    }
    await smRunPitchAsync(samples, sr);

    document.getElementById('sm-wave-wrap').style.display = 'block';
    // Resume after the last transcribed line already in the sheet (e.g. a reloaded song)
    const resume = smLastTranscribedEnd();
    if (resume > 0) document.getElementById('sm-start-sec').value = (resume + SM_SECTION_GAP).toFixed(2);
    smSyncSectionEnd();
    smInitWaveform();
    smDrawWaveform();
    smSetStatus('sm-load-status', `✅ ${file.name} — ${SM.audioBuffer.duration.toFixed(1)}s` +
        (resume > 0 ? ` · resuming after ${resume.toFixed(2)}s` : ''));
}

function smLastTranscribedEnd() {
    let t = 0;
    document.querySelectorAll('.line-container:not(.draft-line)').forEach(l => {
        if (l.dataset.audioEnd !== undefined) t = Math.max(t, parseFloat(l.dataset.audioEnd) || 0);
    });
    return t;
}
async function smRunPitchAsync(samples, sr) {
    const hop    = SM.hopLen;
    const frame  = SM.frameLen;
    const nFrames = Math.floor((samples.length - frame) / hop) + 1;
    const rawPitch = new Float32Array(nFrames);
    const energy   = new Float32Array(nFrames);
    const hann = new Float32Array(frame);
    for (let i = 0; i < frame; i++) hann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (frame - 1)));

    const CHUNK = 200; // frames per async tick
    for (let fi = 0; fi < nFrames; fi += CHUNK) {
        const end = Math.min(fi + CHUNK, nFrames);
        for (let f = fi; f < end; f++) {
            const start = f * hop;
            const buf = new Float32Array(frame);
            let rmsSum = 0;
            for (let i = 0; i < frame; i++) {
                const v = samples[start + i] * hann[i];
                buf[i] = v;
                rmsSum += v * v;
            }
            energy[f]   = Math.sqrt(rmsSum / frame);
            rawPitch[f] = yinDetectFrame(buf, sr); // stores 0 if unvoiced/out-of-range
        }
        // Yield to browser
        await new Promise(r => setTimeout(r, 0));
        const pct = Math.round(100 * end / nFrames);
        if (pct % 10 === 0) smSetStatus('sm-load-status', `⏳ Pitch tracking… ${pct}%`);
    }
    SM.rawPitchTrack = rawPitch;
    SM.energyTrack   = energy;
    SM.pitchTrack    = smBuildPitchTrack(); // apply default gate + smoothing
}

// Build (or rebuild) the gated+smoothed pitch track from the raw data.
// Called once after loading and again at each Detect so gate changes take effect immediately.
function smBuildPitchTrack() {
    if (!SM.rawPitchTrack) return null;
    const rmsGate = parseFloat(document.getElementById('sm-rms-gate')?.value) || 0.010;
    const gated = SM.rawPitchTrack.map((hz, i) =>
        (SM.energyTrack[i] >= rmsGate && hz >= 80 && hz <= 1200) ? hz : 0
    );
    return smSmoothPitch(gated, 17); // wider window suppresses tabla transients
}

// ── Waveform canvas ────────────────────────────────────────────────────────────
function smDrawWaveform() {
    const canvas = document.getElementById('sm-waveform-canvas');
    if (!canvas || !canvas.offsetWidth) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth, H = canvas.height;
    canvas.width = W;
    ctx.fillStyle = '#0a0a1e';
    ctx.fillRect(0, 0, W, H);
    if (!SM.audioBuffer) return;
    const dur = SM.audioBuffer.duration;
    const X = t => (t / dur) * W;

    // 0.5 s time grid
    ctx.font = '9px monospace'; ctx.textAlign = 'left';
    for (let t = 0; t <= dur + 0.01; t += 0.5) {
        const x = Math.round(X(t)), isSec = Math.abs(t - Math.round(t)) < 0.01;
        ctx.strokeStyle = isSec ? 'rgba(100,100,160,0.7)' : 'rgba(60,60,100,0.5)';
        ctx.lineWidth = isSec ? 1 : 0.5;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        if (isSec && t > 0.1) { ctx.fillStyle = 'rgba(140,140,190,0.85)'; ctx.fillText(`${t.toFixed(0)}s`, x + 2, 10); }
    }

    // Waveform envelope
    const samples = SM.audioBuffer.getChannelData(0);
    const step = Math.max(1, Math.floor(samples.length / W));
    ctx.strokeStyle = '#4a4a9a'; ctx.lineWidth = 1; ctx.beginPath();
    for (let x = 0; x < W; x++) {
        let max = 0;
        for (let j = 0; j < step; j++) { const v = Math.abs(samples[x * step + j] || 0); if (v > max) max = v; }
        if (x === 0) ctx.moveTo(x, (1 - max) * H / 2); else ctx.lineTo(x, (1 - max) * H / 2);
        ctx.lineTo(x, (1 + max) * H / 2);
    }
    ctx.stroke();

    // Pitch track overlay
    if (SM.pitchTrack) {
        const spf = SM.hopLen / SM.sampleRate;
        ctx.strokeStyle = '#5858ff'; ctx.lineWidth = 1.5; ctx.beginPath();
        let first = true;
        for (let fi = 0; fi < SM.pitchTrack.length; fi++) {
            const hz = SM.pitchTrack[fi];
            if (hz <= 0) { first = true; continue; }
            const y = H - (Math.log2(hz / 80) / Math.log2(700 / 80)) * H * 0.8 - H * 0.1;
            if (first) { ctx.moveTo(X(fi * spf), y); first = false; } else ctx.lineTo(X(fi * spf), y);
        }
        ctx.stroke();
    }

    // Already transcribed lines (green band at the top)
    ctx.fillStyle = 'rgba(34,204,85,0.55)';
    document.querySelectorAll('.line-container').forEach(l => {
        if (l.dataset.audioStart === undefined) return;
        const a = X(parseFloat(l.dataset.audioStart)), b = X(parseFloat(l.dataset.audioEnd));
        ctx.fillStyle = l.classList.contains('draft-line') ? 'rgba(249,168,37,0.8)' : 'rgba(34,204,85,0.6)';
        ctx.fillRect(a, 0, Math.max(1, b - a - 1), 4);
    });

    // Current section
    const xs = X(SM.segStart), xe = X(SM.segEnd);
    ctx.fillStyle = 'rgba(74,74,255,0.15)'; ctx.fillRect(xs, 0, xe - xs, H);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#4a4aff'; ctx.beginPath(); ctx.moveTo(xs, 0); ctx.lineTo(xs, H); ctx.stroke();
    ctx.strokeStyle = '#ff9944'; ctx.beginPath(); ctx.moveTo(xe, 0); ctx.lineTo(xe, H); ctx.stroke();
    if (SM.segAdlibSec > 0) {   // ad-lib part (orange) + where beat 1 begins
        const xa = X(SM.segStart + SM.segAdlibSec);
        ctx.fillStyle = 'rgba(255,153,68,0.18)'; ctx.fillRect(xs, 0, xa - xs, H);
        ctx.setLineDash([3, 3]); ctx.strokeStyle = '#ffaa44'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(xa, 0); ctx.lineTo(xa, H); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = '#ffaa44'; ctx.font = '10px monospace'; ctx.textAlign = 'left';
        ctx.fillText('ad-lib', xs + 3, H - 6);
    }
    ctx.font = '10px monospace';
    ctx.fillStyle = '#4a9aff'; ctx.textAlign = 'left';  ctx.fillText(`${SM.segStart.toFixed(2)}s`, xs + 3, H / 2 - 4);
    ctx.fillStyle = '#ff9944'; ctx.textAlign = 'right'; ctx.fillText(`${SM.segEnd.toFixed(2)}s`, xe - 3, H / 2 - 4);
    ctx.textAlign = 'left';

    // Playhead while original audio plays
    if (SM.playheadT >= 0 && SM.playheadT <= dur) {
        const xh = Math.round(X(SM.playheadT));
        ctx.shadowColor = '#ffff88'; ctx.shadowBlur = 8;
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(xh, 0); ctx.lineTo(xh, H); ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#ffff88';
        ctx.beginPath(); ctx.moveTo(xh - 5, 0); ctx.lineTo(xh + 5, 0); ctx.lineTo(xh, 9); ctx.closePath(); ctx.fill();
        ctx.font = '10px monospace'; ctx.textAlign = xh > W - 40 ? 'right' : 'left';
        ctx.fillText(`${SM.playheadT.toFixed(2)}s`, xh + (xh > W - 40 ? -4 : 4), 20);
        ctx.textAlign = 'left';
    }
}

function smInitWaveform() {
    const canvas = document.getElementById('sm-waveform-canvas');
    if (!canvas || canvas._smInited) return;
    canvas._smInited = true;
    const tAt = e => (e.offsetX / canvas.offsetWidth) * SM.audioBuffer.duration;
    canvas.addEventListener('mousedown', e => {
        if (!SM.audioBuffer) return;
        SM.dragX = e.offsetX;
        if (e.shiftKey) { SM.dragging = true; smSetLinesFromEndTime(tAt(e)); return; }
        SM.dragging = true;   // plain click sets Start; dragging (> 6 px) afterwards sets the length
        document.getElementById('sm-start-sec').value = +tAt(e).toFixed(2);
        smSyncSectionEnd();
    });
    canvas.addEventListener('mousemove', e => {
        if (SM.dragging && SM.audioBuffer && e.buttons && Math.abs(e.offsetX - SM.dragX) > 6) smSetLinesFromEndTime(tAt(e));
    });
    ['mouseup', 'mouseleave'].forEach(ev => canvas.addEventListener(ev, () => { SM.dragging = false; }));
    window.addEventListener('resize', () => smDrawWaveform());
}

// ── Section timing ─────────────────────────────────────────────────────────────
//   End = Start + Ad-lib + Lines × BeatsPerLine × 60 / BPM
function smSectionLines() { const v = parseInt(document.getElementById('sm-seg-lines').value); return Math.max(0, isNaN(v) ? 1 : v); }
function smAdlibSec()     { return Math.max(0, parseFloat(document.getElementById('sm-adlib-sec').value) || 0); }

// Line layout of a section: optional ad-lib line, then Taal lines.
// Each range: {s, e (beat indices), adlib, t0, t1 (seconds), beatDur}
function smLineRanges(total) {
    const beatDur = 60 / smBpm();
    const nA  = Math.min(SM.segAdlibBeats || 0, total);
    const bpl = SM.segBpl || total || 1;
    const out = [];
    let t = SM.segStart || 0;
    if (nA > 0) {
        out.push({ s: 0, e: nA, adlib: true, t0: t, t1: t + SM.segAdlibSec, beatDur: SM.segAdlibSec / nA });
        t += SM.segAdlibSec;
    }
    for (let s0 = nA; s0 < total; s0 += bpl) {
        const e0 = Math.min(s0 + bpl, total);
        out.push({ s: s0, e: e0, adlib: false, t0: t, t1: t + (e0 - s0) * beatDur, beatDur });
        t += (e0 - s0) * beatDur;
    }
    return out;
}

function smSyncSectionEnd() {
    const bpm   = smBpm();
    const bpl   = smBeatsPerLine();
    const adlib = smAdlibSec();
    let lines   = smSectionLines();
    if (adlib <= 0 && lines < 1) { lines = 1; document.getElementById('sm-seg-lines').value = 1; }
    const start = Math.max(0, parseFloat(document.getElementById('sm-start-sec').value) || 0);
    const beats = lines * bpl;
    const dur   = beats * 60 / bpm;
    const nA    = adlib > 0 ? Math.max(1, Math.round(adlib * bpm / 60)) : 0;   // ad-lib slots, one per beat length
    const end   = start + adlib + dur;
    document.getElementById('sm-end-sec').value = end.toFixed(2);
    Object.assign(SM, { segStart: +start.toFixed(2), segEnd: +end.toFixed(2), segBeats: nA + beats,
                        segBpl: bpl, segAdlibSec: adlib, segAdlibBeats: nA });
    const info = document.getElementById('sm-section-info');
    if (info) {
        let txt, warn = false;
        if (!bpl) { txt = '⚠ Choose a Taal in the toolbar — it sets the beats per line'; warn = true; }
        else {
            txt = (nA ? `Ad-lib ${adlib.toFixed(2)}s (${nA} slots) + ` : '') +
                  `${lines} × ${bpl} = ${beats} beats · ${dur.toFixed(2)}s @ ${bpm} BPM` +
                  (nA ? ` · beat 1 at ${(start + adlib).toFixed(2)}s` : '');
            if (SM.audioBuffer && end > SM.audioBuffer.duration + 0.01) {
                txt += ` · ⚠ ends past the audio (${SM.audioBuffer.duration.toFixed(2)}s)`; warn = true;
            }
        }
        info.textContent = txt;
        info.classList.toggle('warn', warn);
    }
    smDrawWaveform();
}

// Shift-click / drag: whole number of lines closest to time t (counted from beat 1)
function smSetLinesFromEndTime(t) {
    const bpl = smBeatsPerLine();
    if (!bpl) return;
    const lineDur = bpl * 60 / smBpm();
    const beatsAt = (parseFloat(document.getElementById('sm-start-sec').value) || 0) + smAdlibSec();
    document.getElementById('sm-seg-lines').value =
        Math.max(smAdlibSec() > 0 ? 0 : 1, Math.round((t - beatsAt) / lineDur));
    smSyncSectionEnd();
}

// Next section starts where this one ended + gap; ad-lib applies only to the section it was entered for
function smAdvanceToNextSection(fromEnd) {
    document.getElementById('sm-start-sec').value = ((fromEnd ?? SM.segEnd) + SM_SECTION_GAP).toFixed(2);
    document.getElementById('sm-adlib-sec').value = 0;
    if (smSectionLines() < 1) document.getElementById('sm-seg-lines').value = 1;
    smSyncSectionEnd();
}

// ── Detect → draft lines in the song sheet ─────────────────────────────────────
async function smDetect() {
    smSyncSectionEnd();
    if (!SM.audioFile) { smSetStatus('sm-status', '❌ Load an audio file first.'); return; }
    if (!smBeatsPerLine()) { smSetStatus('sm-status', '❌ Choose a Taal in the toolbar first (it sets the beats per line).'); return; }
    if (SM.segBeats < 1) { smSetStatus('sm-status', '❌ Nothing to detect — set Lines or Ad-lib.'); return; }
    let raw;
    if (SMB.enabled()) {
        raw = await SMB.detectSection();
        if (!raw) return;
    } else {
        if (!SM.rawPitchTrack) { smSetStatus('sm-status', '⏳ Audio is still being analysed…'); return; }
        SM.pitchTrack = smBuildPitchTrack();   // current Energy Gate, no reload needed
        const nA = SM.segAdlibBeats;
        raw = (nA ? smProcessSegment(SM.segStart, nA, SM.segAdlibSec / nA) : [])
            .concat(smProcessSegment(SM.segStart + SM.segAdlibSec, SM.segBeats - nA, 60 / smBpm()));
    }
    smDiscardDraft(true);
    const beats = smApplySustain(raw, SM.carryNote, SM.carryOct);
    const created = smInsertDraftLines(beats, smLineRanges(beats.length), null, {});
    SM.draftMode = 'section';
    smShowDraftBar();
    smSetStatus('sm-status', `🔍 ${created.length} draft line(s) added to the sheet — edit them there, then ✔ Accept.`);
    smDrawWaveform();
}

// Instrument for new lines: same as the last line in the sheet, else harmonium
function smDefaultInstrument() {
    const lines = document.querySelectorAll('.line-container:not(.draft-line)');
    const last = lines[lines.length - 1];
    return last ? last.querySelector('.line-inst-select').value : 'harmonium';
}

// beats: [{notes, octs, clusters}] for the whole stretch; ranges from smLineRanges
// opts: {section, instrument, percussion} overrides (used by Re-detect)
function smInsertDraftLines(beats, ranges, insertBefore, opts) {
    const section  = opts.section ?? (document.getElementById('sm-section').value.trim() || 'Line');
    const instr    = opts.instrument || smDefaultInstrument();
    const mute     = document.getElementById('sm-mute-perc').checked;
    const taal     = smTaalKey();
    const created  = [];
    ranges.forEach(rg => {
        const chunk = beats.slice(rg.s, rg.e);
        const el = addLine({
            section, line_instrument: instr,
            percussion: rg.adlib ? 'mute' : (opts.percussion || (mute ? 'mute' : 'tabla')),
            line_volume: 1.0, adlib: rg.adlib, legato: false,
            // Ad-lib slots are spread over the ad-lib time exactly via a fractional Line BPM
            line_bpm: rg.adlib ? +(60 / rg.beatDur).toFixed(3) : null,
            taal_key: taal, beats: String(chunk.length), lyrics: [],
            notes:   chunk.map(b => [...b.notes]),
            octaves: chunk.map(b => [...b.octs]),
            meend:   chunk.map(b => b.notes.map(() => false)),
            audio_start: +rg.t0.toFixed(3), audio_end: +rg.t1.toFixed(3)
        }, insertBefore || undefined);
        el.classList.add('draft-line');
        el.querySelectorAll('.grid-row[data-row="notes"] .cell').forEach((cell, i) => {
            cell._smClusters = (chunk[i] && chunk[i].clusters) || [];
            smRenderChips(cell);
        });
        created.push(el);
    });
    renumberLines();
    applyRagaHighlights();
    if (created[0]) created[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
    return created;
}

function smDraftLines() { return Array.from(document.querySelectorAll('.line-container.draft-line')); }

function smShowDraftBar() {
    const d = smDraftLines();
    if (!d.length) { smHideDraftBar(); return; }
    const t0 = Math.min(...d.map(l => +l.dataset.audioStart)), t1 = Math.max(...d.map(l => +l.dataset.audioEnd));
    const nA = d.filter(l => l.classList.contains('adlib-line')).length;
    document.getElementById('sm-draft-info').textContent =
        `📝 Draft: ${nA ? 'ad-lib + ' : ''}${d.length - nA} line(s) · ${t0.toFixed(2)}–${t1.toFixed(2)}s`;
    document.getElementById('sm-draft-skip').style.display = SM.draftMode === 'section' ? '' : 'none';
    document.getElementById('sm-draft-bar').classList.add('open');
}
function smHideDraftBar() { document.getElementById('sm-draft-bar').classList.remove('open'); }

function smAcceptDraft() {
    const drafts = smDraftLines();
    if (!drafts.length) return;
    stopPlayback(); smStopOriginal();
    drafts.forEach(l => {
        l.classList.remove('draft-line');
        l.querySelectorAll('.sm-cands').forEach(c => c.remove());
    });
    if (SM.draftMode === 'section') {
        // Sustain carry-over into the next section: last sounding note of this one
        const last = lineToData(drafts[drafts.length - 1]);
        for (let b = last.notes.length - 1; b >= 0; b--) {
            const k = last.notes[b].length - 1, n = last.notes[b][k];
            if (n !== '-' && n !== '~') { SM.carryNote = n; SM.carryOct = last.octaves[b][k]; break; }
        }
        smAdvanceToNextSection(Math.max(...drafts.map(l => +l.dataset.audioEnd)));
    }
    SM.replacedLine = null; SM.draftMode = null;
    smHideDraftBar(); renumberLines(); smDrawWaveform();
    smSetStatus('sm-status', `✅ Accepted ${drafts.length} line(s).` + (SM.audioBuffer ? ' Next section is set — Detect when ready.' : ''));
}

// quiet = called internally before a new Detect
function smDiscardDraft(quiet) {
    const drafts = smDraftLines();
    if (drafts.length || SM.replacedLine) { stopPlayback(); smStopOriginal(); }
    if (SM.replacedLine && drafts.length) drafts[0].parentNode.insertBefore(SM.replacedLine, drafts[0]);
    drafts.forEach(l => l.remove());
    SM.replacedLine = null; SM.draftMode = null;
    smHideDraftBar(); renumberLines(); smDrawWaveform();
    if (!quiet) smSetStatus('sm-status', '✖ Draft discarded.');
}

function smSkipSection() {
    const end = SM.segEnd;
    smDiscardDraft(true);
    smAdvanceToNextSection(end);
    smSetStatus('sm-status', '⏭ Section skipped.');
}

function smToggleDraftPlay() {
    if (playbackTimer) { stopPlayback(); return; }
    const d = smDraftLines();
    if (d.length) startPlayback(d);
}
function smPlayDraftOriginal(btn) {
    const d = smDraftLines();
    if (!d.length) return;
    smPlayRange(Math.min(...d.map(l => +l.dataset.audioStart)), Math.max(...d.map(l => +l.dataset.audioEnd)), btn);
}

// ── Re-detect one line in place (uses the line's stored audio times) ──────────
async function smRedetectLine(line) {
    if (!SM.audioFile) {
        smToggleTranscribe(true);
        smSetStatus('sm-status', '❌ Load this song’s audio file first, then ⟲ Re-detect.');
        document.getElementById('sm-panel').scrollIntoView({ behavior: 'smooth' });
        return;
    }
    smDiscardDraft(true);
    const d  = lineToData(line);
    const n  = parseInt(d.beats) || d.notes.length;
    const t0 = d.audio_start, t1 = d.audio_end, beatDur = (t1 - t0) / n;
    let raw;
    if (SMB.enabled()) {
        raw = await SMB.segment(t0, n, 60 / beatDur);
        if (!raw) return;
    } else {
        if (!SM.rawPitchTrack) return;
        SM.pitchTrack = smBuildPitchTrack();
        raw = smProcessSegment(t0, n, beatDur);
    }
    const beats = smApplySustain(raw, null, 0);
    smInsertDraftLines(beats, [{ s: 0, e: n, adlib: d.adlib, t0, t1, beatDur }], line,
        { section: d.section, instrument: d.line_instrument, percussion: d.percussion });
    SM.replacedLine = line;
    line.remove();
    SM.draftMode = 'redetect';
    renumberLines(); smShowDraftBar(); smDrawWaveform();
    smSetStatus('sm-status', '⟲ Line re-detected as a draft — ✔ Accept to keep it, ✖ Discard to restore the old line.');
}

// ── Original-audio playback (section, draft, or one line) with waveform playhead ──
function smPlayRange(t0, t1, btn) {
    const same = SM.play && SM.play.btn === btn;
    smStopOriginal();
    if (same) return;   // second click = stop
    if (!SM.audioBuffer) { smSetStatus('sm-status', '❌ Load the song’s audio file to play the original.'); return; }
    if (SM.audioCtx.state === 'suspended') SM.audioCtx.resume();
    const src = SM.audioCtx.createBufferSource();
    src.buffer = SM.audioBuffer;
    src.connect(SM.audioCtx.destination);
    const dur = Math.max(0.1, t1 - t0);
    src.start(0, t0, dur);
    src.onended = () => { if (SM.play && SM.play.src === src) smStopOriginal(); };
    SM.play = { src, startCtx: SM.audioCtx.currentTime, t0, dur, btn };
    if (btn) { btn.dataset.label = btn.dataset.label || btn.textContent; btn.textContent = '⏹ Stop'; btn.classList.add('playing'); }
    smAnimatePlayhead();
}
function smStopOriginal() {
    if (SM.animFrameId) { cancelAnimationFrame(SM.animFrameId); SM.animFrameId = null; }
    if (SM.play) {
        try { SM.play.src.stop(); } catch (e) {}
        const b = SM.play.btn;
        if (b) { b.textContent = b.dataset.label || b.textContent; b.classList.remove('playing'); }
        SM.play = null;
    }
    SM.playheadT = -1;
    smDrawWaveform();
}
function smAnimatePlayhead() {
    if (!SM.play) return;
    const el = SM.audioCtx.currentTime - SM.play.startCtx;
    SM.playheadT = SM.play.t0 + el;
    smDrawWaveform();
    if (el < SM.play.dur + 0.05) SM.animFrameId = requestAnimationFrame(smAnimatePlayhead);
}
function smToggleSectionOriginal(btn) { smSyncSectionEnd(); smPlayRange(SM.segStart, SM.segEnd, btn); }

// ── Pitch-candidate chips (draft lines, single-note beats) ────────────────────
function smOctMark(o) { return o > 0 ? "'".repeat(o) : (o < 0 ? ",".repeat(-o) : ''); }
function smRenderChips(cell) {
    const old = cell.querySelector('.sm-cands');
    if (old) old.remove();
    if (!cell.closest('.draft-line')) return;
    const units = cell.querySelectorAll('.note-unit');
    if (units.length !== 1) return;
    const note = units[0].querySelector('.note-select').value;
    if (note === '-' || note === '~') return;
    const ob  = units[0].querySelector('.octave-btn.active');
    const oct = ob ? parseInt(ob.dataset.oct) : 0;
    const cands = [];
    const idx = SM_SWARA_NAMES.indexOf(note);
    if (idx >= 0) {   // komal/shuddha neighbours and octave alternates of the current note
        const base = oct * 12 + idx;
        [0, -1, 1, 12, -12].forEach(d => { const s = base + d;
            cands.push({ note: SM_SWARA_NAMES[((s % 12) + 12) % 12], oct: Math.floor(s / 12) }); });
    }
    (cell._smClusters || []).forEach(c => { if (c.note && c.note !== '-') cands.push({ note: c.note, oct: c.oct, pct: c.pct }); });
    const seen = {}, uniq = [];
    cands.forEach(c => { const k = c.note + ':' + c.oct; if (!seen[k] && c.oct >= -2 && c.oct <= 2) { seen[k] = 1; uniq.push(c); } });
    const div = document.createElement('div');
    div.className = 'sm-cands';
    div.title = 'Likely pitches — click the one you hear';
    div.innerHTML = uniq.slice(0, 6).map(c => {
        const cur = c.note === note && c.oct === oct;
        const komal = /^[a-z]/.test(c.note);
        return `<button type="button" class="sm-cand${cur ? ' cur' : ''}${komal ? ' komal' : ''}" data-note="${c.note}" data-oct="${c.oct}">` +
               `${c.note}${smOctMark(c.oct)}${c.pct ? `<i>${c.pct}%</i>` : ''}</button>`;
    }).join('');
    cell.appendChild(div);
}
function smApplyChip(chip) {
    const cell = chip.closest('.cell');
    const unit = cell.querySelector('.note-unit');
    const oct  = parseInt(chip.dataset.oct);
    unit.querySelector('.note-select').value = chip.dataset.note;
    unit.querySelectorAll('.octave-btn').forEach(b => b.classList.toggle('active', oct !== 0 && parseInt(b.dataset.oct) === oct));
    applySustainStyle(unit);
    applyRagaHighlights();
    smRenderChips(cell);
}

// ── Composer hook: per-line audio tag (time, ▶ Orig, ⟲ Re-detect) ────────────
function onLineCreated(line, data) {
    if (line.dataset.audioStart === undefined) return;
    const a = parseFloat(line.dataset.audioStart), b = parseFloat(line.dataset.audioEnd);
    const tag = document.createElement('span');
    tag.className = 'line-audio';
    tag.title = 'Position of this line in the recording';
    tag.innerHTML = `🎤 ${a.toFixed(2)}–${b.toFixed(2)}s` +
        `<button type="button" class="sm-line-orig" title="Play this part of the original recording">▶ Orig</button>` +
        `<button type="button" class="sm-line-redetect" title="Detect this line again from the recording">⟲ Re-detect</button>`;
    line.classList.add('has-audio');
    line.appendChild(tag);
}

// ── AI cleanup of draft lines (uses the model + key set in ✨ AI Compose) ─────
async function smAICleanup() {
    const drafts = smDraftLines();
    if (!drafts.length) return;
    const model = document.getElementById('ai-model')?.value;
    if (!model) { smSetStatus('sm-status', '❌ No AI model available.'); return; }
    const provider = aiGetProvider(model);
    const key = (document.getElementById('ai-api-key')?.value || '').trim() ||
                localStorage.getItem(`ai_key_${provider}`) || localStorage.getItem('gemini_api_key') || '';
    if (!key) { smSetStatus('sm-status', '❌ Open ✨ AI Compose once, choose a model and save its API key.'); return; }
    const data = drafts.map(lineToData);
    const slim = data.map(d => ({ adlib: d.adlib, notes: d.notes, octaves: d.octaves }));
    const prompt = `You are a Hindustani music expert. Review this auto-detected sargam notation and correct errors.
Sa = ${smScaleSa().toFixed(2)} Hz. Raga: ${document.getElementById('raga-select')?.value || 'unknown'}.
Note names: Sa re Re ga Ga Ma ma Pa dha Dha ni Ni (lowercase = komal, "ma" = tivra), "-" = rest, "~" = sustain.
Octaves: 0 = madhya, 1/2 = taar, -1/-2 = mandra.
Rules:
1. Fix obvious pitch-detection errors using melodic context and natural phrase flow.
2. Correct octave jumps that are clearly detection errors.
3. Keep changes minimal; do not invent notes the input does not suggest.
4. Keep EXACTLY the same number of lines, the same number of beats per line, and the same number of notes inside each beat.
5. Return ONLY JSON: {"lines":[{"notes":[[...]],"octaves":[[...]]}, ...]}
Input:
${JSON.stringify({ lines: slim })}`;
    smSetStatus('sm-status', '⏳ AI cleanup…');
    try {
        const text = await aiCallLLM(prompt, model, key, { temperature: 0.2, maxTokens: 8192 });
        const res  = JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1)).lines;
        const valid = new Set(['-', '~', ...SM_SWARA_NAMES]);
        let changed = 0;
        drafts.forEach((el, i) => {
            const r = res && res[i];
            if (!r || !Array.isArray(r.notes) || r.notes.length !== data[i].notes.length) return;
            const ok = r.notes.every((bn, bi) => Array.isArray(bn) && bn.length === data[i].notes[bi].length && bn.every(n => valid.has(n)));
            if (!ok) return;
            const octs = r.notes.map((bn, bi) => bn.map((_, k) => {
                const o = parseInt(r.octaves?.[bi]?.[k]); return isNaN(o) ? 0 : Math.max(-2, Math.min(2, o)); }));
            const clusters = Array.from(el.querySelectorAll('.grid-row[data-row="notes"] .cell')).map(c => c._smClusters);
            const ne = addLine({ ...data[i], notes: r.notes, octaves: octs }, el);
            ne.classList.add('draft-line');
            ne.querySelectorAll('.grid-row[data-row="notes"] .cell').forEach((c, k) => { c._smClusters = clusters[k] || []; smRenderChips(c); });
            el.remove();
            changed++;
        });
        renumberLines(); applyRagaHighlights();
        smSetStatus('sm-status', changed ? `✅ AI cleanup applied to ${changed} draft line(s) — review, then ✔ Accept.`
                                         : '⚠ AI reply did not match the draft shape; nothing changed.');
    } catch (e) {
        smSetStatus('sm-status', `❌ AI error: ${e.message}`);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  AI backend (CREPE) — when "AI backend" is ticked, detection uses the local
//  FastAPI service (server.py) instead of the in-browser YIN. Same beat shape.
// ═══════════════════════════════════════════════════════════════════════════
const SMB = {
  enabled() { const c = document.getElementById('sm-use-backend'); return !!(c && c.checked); },
  cfg() {
    return {
      url: (document.getElementById('sm-backend-url').value || 'http://127.0.0.1:8000').replace(/\/+$/, ''),
      engine: (document.getElementById('sm-backend-engine') || {}).value || 'crepe',
      model: document.getElementById('sm-backend-model').value,
      demucs: document.getElementById('sm-backend-demucs').checked,
      device: 'cpu',   // set to 'auto' or 'cuda' once a CUDA torch build is installed
    };
  },
  status(m) { smSetStatus('sm-backend-status', m); },
  async poll(url, jobId, onprog) {
    for (;;) {
      await new Promise(r => setTimeout(r, 700));
      const j = await (await fetch(url + '/job/' + jobId)).json();
      if (onprog && typeof j.progress === 'number') onprog(j.progress, j.message || '');
      if (j.status === 'done') return j.result;
      if (j.status === 'error') throw new Error(j.error || 'job failed');
    }
  },
  async openSession(file) {
    const c = this.cfg();
    try {
      SM.sessionId = null; SM.sessionReady = false;
      this.status('⏳ opening on backend…');
      const fd = new FormData();
      fd.append('file', file); fd.append('demucs', c.demucs ? 'true' : 'false'); fd.append('device', c.device);
      const r = await fetch(c.url + '/session', { method: 'POST', body: fd });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const { job_id } = await r.json();
      const res = await this.poll(c.url, job_id, (p, m) => this.status((m || 'preparing') + ' ' + Math.round(p * 100) + '%'));
      SM.sessionId = res.session_id; SM.sessionReady = true;
      this.status('✅ ready' + (res.demucs_used ? ' (vocal separated)' : '') + ' — ' + res.duration.toFixed(1) + 's');
    } catch (e) { this.status('❌ backend: ' + e.message + ' (is the server running?)'); }
  },
  async ensureSession() {
    if (!SM.audioFile) { smSetStatus('sm-status', '❌ Load an audio file first.'); return false; }
    if (!SM.sessionId) { this.status('⏳ backend not ready — opening…'); await this.openSession(SM.audioFile); }
    return !!SM.sessionId;
  },
  async analyze() {
    if (!this.enabled()) { this.status('tick "AI backend" first'); return; }
    if (!await this.ensureSession()) return;
    const c = this.cfg();
    try {
      this.status('⏳ pre-analyzing whole song…');
      const fd = new FormData();
      fd.append('session_id', SM.sessionId); fd.append('model', c.model); fd.append('device', c.device); fd.append('engine', c.engine);
      const r = await fetch(c.url + '/analyze', { method: 'POST', body: fd });
      const { job_id } = await r.json();
      await this.poll(c.url, job_id, (p) => this.status('⏳ pre-analyzing ' + Math.round(p * 100) + '%'));
      this.status('✅ pre-analyzed — detections are now instant');
    } catch (e) { this.status('❌ ' + e.message); }
  },
  // One stretch: `beats` cells of 60/bpm seconds from `start`. Returns beat objects or null.
  async segment(start, beats, bpm) {
    if (!await this.ensureSession()) return null;
    const c = this.cfg();
    try {
      const fd = new FormData();
      fd.append('session_id', SM.sessionId);
      fd.append('start', start); fd.append('beats', beats);
      fd.append('bpm', bpm); fd.append('sa', smDetectSa());
      fd.append('model', c.model); fd.append('device', c.device); fd.append('engine', c.engine);
      const r = await fetch(c.url + '/segment', { method: 'POST', body: fd });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      this.status('✅ ' + beats + ' beats @ ' + (+start).toFixed(2) + 's (Sa ' + data.sa + ' Hz)');
      return data.beats.map(b => ({ notes: b.notes, octs: b.octs, dom_hz: 0, voiced_pct: b.voiced_pct || 0, clusters: [] }));
    } catch (e) {
      smSetStatus('sm-status', '❌ backend: ' + e.message); this.status('❌ ' + e.message);
      return null;
    }
  },
  async detectSection() {
    smSetStatus('sm-status', '🔍 Detecting via backend…');
    const nA = SM.segAdlibBeats || 0, nT = SM.segBeats - nA;
    let cells = [];
    if (nA) { const a = await this.segment(SM.segStart, nA, 60 * nA / SM.segAdlibSec); if (!a) return null; cells = cells.concat(a); }
    if (nT) { const t = await this.segment(SM.segStart + SM.segAdlibSec, nT, smBpm()); if (!t) return null; cells = cells.concat(t); }
    return cells;
  }
};

// capacity (full/tiny) only means anything for the CREPE engine -- our
// trained model always runs its own fixed architecture, so hide that
// dropdown when "Our Trained Model" is selected.
function smOnBackendEngineChange() {
  const engineEl = document.getElementById('sm-backend-engine');
  const modelEl = document.getElementById('sm-backend-model');
  if (!engineEl || !modelEl) return;
  modelEl.style.display = (engineEl.value === 'model') ? 'none' : '';
}

// Picking a Backend URL sets a sensible default engine for it: the cloud
// trained-model service only ever runs that one engine, while the local
// server.py needs to actually be started (start_backend.bat) before CREPE
// (or the model) will respond. The engine dropdown stays changeable
// afterward -- this just sets a sane starting point per backend.
function smOnBackendUrlChange() {
  const urlEl = document.getElementById('sm-backend-url');
  const engineEl = document.getElementById('sm-backend-engine');
  const noteEl = document.getElementById('sm-backend-note');
  if (!urlEl || !engineEl) return;
  const isLocal = urlEl.value.indexOf('127.0.0.1') !== -1 || urlEl.value.indexOf('localhost') !== -1;
  engineEl.value = isLocal ? 'crepe' : 'model';
  if (noteEl) {
    noteEl.textContent = isLocal
      ? '⚠ Start the local backend first: run start_backend.bat in the project folder, then reload this page.'
      : '';
  }
  smOnBackendEngineChange();
}

// ── Wiring ─────────────────────────────────────────────────────────────────────
(function wireTranscribe() {
  smOnBackendUrlChange();
  // Section inputs + the shared toolbar settings that change timing
  ['sm-start-sec', 'sm-adlib-sec', 'sm-seg-lines', 'bpm', 'taal-select'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input',  smSyncSectionEnd);
    el.addEventListener('change', smSyncSectionEnd);
  });
  const sheet = document.getElementById('song-sheet');
  sheet.addEventListener('click', e => {
    const chip = e.target.closest('.sm-cand');
    if (chip) { smApplyChip(chip); return; }
    const orig = e.target.closest('.sm-line-orig');
    if (orig) {
      const l = orig.closest('.line-container');
      smPlayRange(parseFloat(l.dataset.audioStart), parseFloat(l.dataset.audioEnd), orig);
      return;
    }
    const red = e.target.closest('.sm-line-redetect');
    if (red) { smRedetectLine(red.closest('.line-container')); return; }
    // Octave / add / remove sub-note inside a draft → refresh that beat's chips
    if (e.target.closest('.draft-line') && e.target.closest('.octave-btn, .add-sub-btn, .rem-sub-btn')) {
      const cell = e.target.closest('.cell');
      setTimeout(() => cell && cell.isConnected && smRenderChips(cell), 0);
    }
  });
  sheet.addEventListener('change', e => {
    if (e.target.classList.contains('note-select') && e.target.closest('.draft-line'))
      smRenderChips(e.target.closest('.cell'));
  });
  // Removing a line with × keeps the waveform's transcribed-lines band current
  new MutationObserver(() => { if (SM.audioBuffer) smDrawWaveform(); }).observe(sheet, { childList: true });
  let on = false;
  try { on = localStorage.getItem('sl_transcribe') === '1'; } catch (e) {}
  smToggleTranscribe(on);
})();
