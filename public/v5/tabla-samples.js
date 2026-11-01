// ─────────────────────────────────────────────────────────────────────────
// Real tabla bol samples (replaces oscillator synthesis for the main strokes)
//
// Source recordings — all Creative Commons 0 (public domain; free for any
// use, including commercial, with no permission or attribution required):
//   Dha, Dhin  -> "Da_Test05.wav" / "Da_Test11.wav" by subodh3012 (CompMusic
//                 research corpus, Universitat Pompeu Fabra, Barcelona)
//                 https://freesound.org/people/subodh3012/
//   Na, Tin,
//   Ta, Te,
//   Ka, Ge     -> "tabla bols" pack by mmiron
//                 https://freesound.org/people/mmiron/packs/8162/
// License text: https://creativecommons.org/publicdomain/zero/1.0/
//
// Files live in public/samples/Tabla/. If a file fails to load (offline,
// renamed, etc.) playback for that bol silently falls back to the original
// synthesized tone in engine.js — nothing throws.
// ─────────────────────────────────────────────────────────────────────────

const TABLA_SAMPLE_FILES = {
    Dha:  'samples/Tabla/Dha.mp3',
    Dhin: 'samples/Tabla/Dhin.mp3',
    Na:   'samples/Tabla/Na.mp3',
    Tin:  'samples/Tabla/Tin.mp3',
    Ta:   'samples/Tabla/Ta.mp3',
    Te:   'samples/Tabla/Te.mp3',
    Ka:   'samples/Tabla/Ka.mp3',
    Ge:   'samples/Tabla/Ge.mp3'
};

// Every bol token that appears anywhere in TaalBols / DholakBols maps onto
// one of the recorded samples above. Several bols share a sample because no
// free, separately-recorded take exists yet for that exact stroke — this is
// a close substitute, not a perfect one, and can be replaced file-by-file
// later without touching this mapping's shape.
const TABLA_BOL_TO_SAMPLE = {
    Dha: 'Dha', DhaGe: 'Dha', Dhi: 'Dha',
    Dhin: 'Dhin', Dhinna: 'Dhin', Din: 'Dhin',
    Na: 'Na', Gana: 'Na', Tu: 'Na', Tun: 'Na',
    Tin: 'Tin', Tina: 'Tin',
    Ti: 'Te', Hi: 'Te',
    Ta: 'Ta', Tita: 'Ta', Kata: 'Ta',
    Ka: 'Ka', Kat: 'Ka', Kita: 'Ka',
    Ge: 'Ge', Gi: 'Ge', Da: 'Ge', Gadi: 'Ge'
};

const tablaBuffers = {};
let tablaSamplesLoading = false;

// Kicks off (once) fetching + decoding every sample file into an AudioBuffer.
// Safe to call repeatedly — only the first call with a ready audioCtx does
// anything. Call this from initAudio() so buffers are warm before the first
// beat plays.
function loadTablaSamples() {
    if (!audioCtx || tablaSamplesLoading) return;
    tablaSamplesLoading = true;
    Object.keys(TABLA_SAMPLE_FILES).forEach(function (key) {
        fetch(TABLA_SAMPLE_FILES[key])
            .then(function (r) { return r.arrayBuffer(); })
            .then(function (buf) { return audioCtx.decodeAudioData(buf); })
            .then(function (decoded) { tablaBuffers[key] = decoded; })
            .catch(function () { /* leave unset -> engine.js falls back to synthesis */ });
    });
}

// Plays the real sample for `bol` (if loaded) through destNode at time
// `when` (audioCtx.currentTime-relative seconds), pitched by `rate` and
// scaled by `gainVal`. Returns true if it actually scheduled a sample, so
// the caller knows whether to also run the synthesized fallback.
function playTablaSample(bol, when, rate, gainVal, destNode) {
    const sampleKey = TABLA_BOL_TO_SAMPLE[bol];
    if (!sampleKey) return false;
    const buf = tablaBuffers[sampleKey];
    if (!buf) return false;
    const src = audioCtx.createBufferSource();
    const g   = audioCtx.createGain();
    src.buffer = buf;
    src.playbackRate.value = rate > 0 ? rate : 1.0;
    g.gain.value = gainVal;
    src.connect(g);
    g.connect(destNode);
    src.start(when);
    return true;
}

// A real multi-hit Tirkit roll, sequencing quick alternating sample hits
// instead of the old synthesized click pattern.
function playTirkitSample(when, rate, gainVal, destNode) {
    const bufA = tablaBuffers['Ka'];
    const bufB = tablaBuffers['Te'];
    if (!bufA || !bufB) return false;
    [0, 0.07, 0.14, 0.21].forEach(function (t, i) {
        const buf = i % 2 === 0 ? bufA : bufB;
        const src = audioCtx.createBufferSource();
        const g   = audioCtx.createGain();
        src.buffer = buf;
        src.playbackRate.value = (rate > 0 ? rate : 1.0) * 1.15;
        g.gain.value = gainVal * 0.8;
        src.connect(g);
        g.connect(destNode);
        src.start(when + t);
    });
    return true;
}
