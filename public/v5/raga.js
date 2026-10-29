// Swar-Laya Studio v5 — raga: raga list/API, raga mode highlighting, pakkad playback
// (split from the single-file studio.html; scripts load in order and share globals)

// ─── Sustain (hold) helpers ───────────────────────────────────────────────────

// ─── Raga Dropdown ───────────────────────────────────────────────────────────
// Rebuild the raga <select> from the current RagaData (called after load/seed).
function populateRagaDropdown() {
    const sel = document.getElementById('raga-select');
    const prev = sel.value;
    sel.innerHTML = '<option value="">— choose —</option>';

    if (isPremium || isAdmin) {
        // Premium / Admin: show all ragas grouped by thaat
        const thaatMap = {};
        Object.entries(RagaData).forEach(([key, r]) => {
            const t = r.thaat || 'Other';
            if (!thaatMap[t]) thaatMap[t] = [];
            thaatMap[t].push({ key, ...r });
        });
        // Sort thaats alphabetically, ragas within each thaat by order
        Object.keys(thaatMap).sort().forEach(thaat => {
            const grp = document.createElement('optgroup');
            grp.label = thaat + ' Thaat';
            thaatMap[thaat]
                .sort((a, b) => (a.order || 0) - (b.order || 0))
                .forEach(r => {
                    const opt = document.createElement('option');
                    opt.value = r.key;
                    opt.textContent = r.name;
                    grp.appendChild(opt);
                });
            sel.appendChild(grp);
        });
    } else {
        // Free user: show only the curated top 10 popular ragas (matched by name)
        const FREE_RAGA_NAMES = [
            'Bhoopali', 'Bihag', 'Durga', 'Yaman', 'Sarang',
            'Bhairav', 'Bhairavi', 'Malkauns', 'Puriya', 'Asawari'
        ];
        const allRagas = Object.entries(RagaData).map(([key, r]) => ({ key, ...r }));
        const top10 = FREE_RAGA_NAMES
            .map(name => allRagas.find(r => r.name.toLowerCase() === name.toLowerCase()))
            .filter(Boolean);  // skip any not found in the current DB
        const grp = document.createElement('optgroup');
        grp.label = '⭐ Popular Ragas';
        top10.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.key;
            opt.textContent = r.name;
            grp.appendChild(opt);
        });
        sel.appendChild(grp);
        // Upgrade prompt
        const lockOpt = document.createElement('option');
        lockOpt.disabled = true;
        lockOpt.textContent = `🔒 Upgrade for all ${Object.keys(RagaData).length}+ ragas`;
        sel.appendChild(lockOpt);
    }

    if (prev) sel.value = prev;
}

// ── Raga list (Firestore 'ragas' collection) ─────────────────────────────────
// Same collection + rules already used by index.html/v4.html on Firebase
// Hosting (public read, admin-only write via firestore.rules) -- no separate
// Postgres/FastAPI layer. The built-in SEED_RAGAS list is shown immediately
// so the dropdown isn't empty while Firestore responds, then replaced.
async function loadRagasFromFirestore() {
    await null;  // let the rest of the script declare isPremium/isAdmin first
    if (!Object.keys(RagaData).length) {
        RagaData = Object.assign({}, SEED_RAGAS);
        populateRagaDropdown();
    }
    if (!db) return;
    try {
        const snap = await db.collection('ragas').orderBy('order').get();
        const loaded = {};
        snap.forEach(doc => { loaded[doc.id] = doc.data(); });
        if (Object.keys(loaded).length) {
            RagaData = loaded;
            console.log(`Loaded ${Object.keys(RagaData).length} ragas from Firestore.`);
        }
    } catch (e) {
        console.warn('Could not load ragas from Firestore; using the built-in raga list.', e.message);
    }
    populateRagaDropdown();
}

// ─── Raga Mode ────────────────────────────────────────────────────────────────
// Apply vadi/samvadi/out-of-raga highlight classes to all note-units in the sheet.
function applyRagaHighlights() {
    const toggle  = document.getElementById('raga-mode-toggle');
    const ragaSel = document.getElementById('raga-select');
    const ragaKey = ragaSel ? ragaSel.value : '';
    const modeOn  = toggle && toggle.checked && ragaKey;

    document.querySelectorAll('.note-unit').forEach(unit => {
        unit.classList.remove('raga-out', 'raga-vadi', 'raga-samvadi');
        if (!modeOn) return;
        const sw   = unit.querySelector('.note-select').value;
        if (sw === '-' || sw === '~') return;
        const raga = RagaData[ragaKey];
        if (!raga) return;
        if (sw === raga.vadi)                unit.classList.add('raga-vadi');
        else if (sw === raga.samvadi)        unit.classList.add('raga-samvadi');
        else if (!raga.allowed.includes(sw)) unit.classList.add('raga-out');
    });
    updateRagaInfoBar(modeOn ? ragaKey : null);
}

function updateRagaInfoBar(ragaKey) {
    const bar = document.getElementById('raga-info-bar');
    if (!ragaKey || !RagaData[ragaKey]) { bar.style.display = 'none'; return; }
    const raga = RagaData[ragaKey];
    bar.style.display = 'flex';
    document.getElementById('raga-bar-name').textContent    = raga.name;
    document.getElementById('raga-bar-aroha').textContent   = 'Aroha: '   + raga.aroha;
    document.getElementById('raga-bar-avaroha').textContent = 'Avaroha: ' + raga.avaroha;
    document.getElementById('raga-bar-vadi').textContent    = raga.vadi;
    document.getElementById('raga-bar-samvadi').textContent = raga.samvadi;

    // Build Time / Jaati / Rasa badges
    const badges = document.getElementById('raga-info-badges');
    badges.innerHTML = '';
    if (raga.time)  badges.innerHTML += `<span class="raga-badge raga-badge-time">🕐 ${raga.time}</span>`;
    if (raga.jaati) badges.innerHTML += `<span class="raga-badge raga-badge-jaati">♩ ${raga.jaati}</span>`;
    if (raga.rasa)  badges.innerHTML += `<span class="raga-badge raga-badge-rasa">✦ ${raga.rasa}</span>`;

    // Build pakkad chips (replacing the old 12-swar palette)
    const chips = document.getElementById('raga-pakkad-chips');
    chips.innerHTML = '';
    (raga.pakkad || []).forEach(entry => {
        // Support both {note,oct} (Firestore format) and [note,oct] (legacy)
        const sw  = Array.isArray(entry) ? entry[0] : entry.note;
        const oct = Array.isArray(entry) ? entry[1] : entry.oct;
        const chip = document.createElement('span');
        chip.className = 'raga-pakkad-chip';
        // Upper octave → combining dot above; lower → combining dot below
        let label = sw;
        if (oct === 2)  label = sw + '\u0307\u0307'; // two dots above = Taar+1
        if (oct === 1)  label = sw + '\u0307';        // dot above = Taar
        if (oct === -1) label = sw + '\u0323';        // dot below = Mandra
        chip.textContent = label;
        if      (sw === raga.vadi)          chip.classList.add('raga-swar-vadi');
        else if (sw === raga.samvadi)       chip.classList.add('raga-swar-samvadi');
        else if ((raga.allowed||[]).includes(sw)) chip.classList.add('raga-swar-in');
        else                                chip.classList.add('raga-swar-out');
        chips.appendChild(chip);
    });
}

// ─── Pakkad continuous playback ──────────────────────────────────────────────
// All notes are pre-scheduled using Web Audio API timestamps so they play
// as a smooth, gapless stream without JS-timer jitter.

let pakkadSources  = [];   // BufferSourceNode / OscillatorNode handles for cancel
let pakkadEndTimer = null; // single setTimeout for UI reset after last note

// Schedule one pakkad note at a specific audioCtx time.
function schedulePakkadNote(freq, inst, startTime, duration) {
    const buf = findBestSample(inst, freq);
    if (buf && sampleBus) {
        // ── Sample path ──────────────────────────────────────────────────────
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        const rate = freq / buf.baseHz;
        src.playbackRate.setValueAtTime(rate, startTime);

        const env = audioCtx.createGain();
        const t   = startTime;
        if (inst === 'piano' || inst === 'sitar') {
            const pk = 0.55;
            env.gain.setValueAtTime(0,   t);
            env.gain.linearRampToValueAtTime(pk,    t + 0.008);
            env.gain.exponentialRampToValueAtTime(0.001, t + Math.min(duration * 1.5, 2.2));
        } else if (inst === 'flute') {
            const att = 0.07, rel = 0.06, pk = 0.50;
            const sus = Math.max(0.01, duration - att - rel);
            env.gain.setValueAtTime(0,       t);
            env.gain.linearRampToValueAtTime(pk * 0.7, t + att * 0.5);
            env.gain.linearRampToValueAtTime(pk,       t + att);
            env.gain.setValueAtTime(pk,                t + att + sus);
            env.gain.linearRampToValueAtTime(0.001,    t + att + sus + rel);
        } else {
            // harmonium / violin — short attack, long sustain, very brief release
            const att = 0.020, rel = 0.045, pk = 0.50;
            const sus = Math.max(0.01, duration - att - rel);
            env.gain.setValueAtTime(0,    t);
            env.gain.linearRampToValueAtTime(pk,    t + att);
            env.gain.setValueAtTime(pk,             t + att + sus);
            env.gain.linearRampToValueAtTime(0.001, t + att + sus + rel);
        }
        src.connect(env);
        env.connect(sampleBus);
        src.start(startTime);
        src.stop(startTime + duration + 0.20);
        pakkadSources.push(src);
        return;
    }

    // ── Oscillator fallback (samples not yet loaded) ─────────────────────────
    // Create a self-contained per-note oscillator set (no shared mainGain).
    const hSpecs  = InstrumentHarmonics[inst] || InstrumentHarmonics.harmonium;
    const oscType = InstrumentOscType[inst]   || 'sine';
    const fSpec   = InstrumentFilter[inst]    || InstrumentFilter.harmonium;

    const filt = audioCtx.createBiquadFilter();
    filt.type            = fSpec.type;
    filt.frequency.value = fSpec.freq;
    filt.Q.value         = fSpec.Q;

    const env = audioCtx.createGain();
    filt.connect(env);
    env.connect(masterBus);

    const t   = startTime;
    const att = 0.030, rel = 0.050, pk = 0.18;
    const sus = Math.max(0.01, duration - att - rel);
    env.gain.setValueAtTime(0,    t);
    env.gain.linearRampToValueAtTime(pk,     t + att);
    env.gain.setValueAtTime(pk,              t + att + sus);
    env.gain.linearRampToValueAtTime(0.0001, t + att + sus + rel);

    for (const spec of hSpecs) {
        const osc = audioCtx.createOscillator();
        const g   = audioCtx.createGain();
        osc.type         = oscType;
        osc.frequency.value = Math.max(freq * spec.ratio, 1.0);
        osc.detune.value = spec.detuneCents || 0;
        g.gain.value     = spec.gainFactor;
        osc.connect(g);
        g.connect(filt);
        osc.start(startTime);
        osc.stop(startTime + duration + 0.20);
        pakkadSources.push(osc);
    }
}

// Parse an aroha/avaroha string ("Sa Re Ga ma Pa Dha Ni Ṡa") into
// [{note, oct}] entries compatible with schedulePakkadNote.
//
// Handles multiple Unicode encodings used by different data sources:
//  Upper octave:  Ṡ/ṡ (U+1E60/61 precomposed), combining dot above (U+0307),
//                 precomposed 'ȧ' (U+0227) e.g. 'Sȧ' = upper Sa
//  Lower octave:  combining dot below (U+0323), precomposed 'ị' (U+1ECB) e.g. 'Nị',
//                 modifier low vertical line prefix (U+02CC) e.g. 'ˌNi'
function parseRagaString(str) {
    if (!str) return [];
    return str.trim().split(/\s+/).map(token => {
        if (!token) return null;
        // ── Upper octave markers ──────────────────────────────────────────────
        // Ṡ/ṡ (U+1E60 / U+1E61) precomposed = upper Sa
        if (/[\u1E60\u1E61]/u.test(token)) return { note: 'Sa', oct: 1 };
        // Combining dot ABOVE (U+0307) anywhere in token = upper octave
        if (token.includes('\u0307')) return { note: token.replace(/\u0307/g, ''), oct: 1 };
        // Precomposed 'ȧ' (U+0227, Latin small a with dot above) = upper octave
        // e.g. 'Sȧ' (batch-research encoding of upper Sa)
        if (token.includes('\u0227')) return { note: token.replace(/\u0227/g, 'a'), oct: 1 };
        // ── Lower octave markers ──────────────────────────────────────────────
        // Combining dot BELOW (U+0323) = lower octave
        if (token.includes('\u0323')) return { note: token.replace(/\u0323/g, ''), oct: -1 };
        // Precomposed 'ị' (U+1ECB, i with dot below) = lower octave, e.g. 'Nị'
        if (token.includes('\u1ECB')) return { note: token.replace(/\u1ECB/g, 'i'), oct: -1 };
        // Modifier letter low vertical line prefix (U+02CC) = lower octave, e.g. 'ˌNi'
        if (token.startsWith('\u02CC')) return { note: token.slice(1), oct: -1 };
        return { note: token, oct: 0 };
    }).filter(e => e && SwaraMap[e.note] !== undefined && SwaraMap[e.note] !== null);
}

function playPakkad() {
    const ragaKey = document.getElementById('raga-select').value;
    if (!ragaKey || !RagaData[ragaKey]) return;
    const raga = RagaData[ragaKey];

    // Cancel any previous pakkad playback
    stopPakkad();

    initAudio();
    const baseSa  = parseFloat(document.getElementById('scale-select').value);
    const bpm     = parseInt(document.getElementById('bpm').value) || 90;
    // stepSec  = time between successive note starts
    // noteSec  = individual note duration — slightly longer so notes overlap → smooth feel
    const stepSec = (60.0 / bpm) * 0.85;
    const noteSec = stepSec * 1.25;
    const gapSec  = (60.0 / bpm) * 1.2;   // ~1 beat silence between sections

    // Use the raga-mode instrument selector if present, else first line's, else harmonium
    const instEl = document.getElementById('raga-instrument') ||
                   document.querySelector('.line-inst-select');
    const inst   = instEl ? instEl.value : 'harmonium';

    ensureSamplesLoaded(inst);

    const btn = document.getElementById('raga-pakkad-play');
    if (btn) { btn.disabled = true; btn.textContent = '■ Stop'; btn.onclick = stopPakkad; }

    // Build the three sections
    const arohaEntries   = parseRagaString(raga.aroha);
    const avarohaEntries = parseRagaString(raga.avaroha);
    const pakkadEntries  = (raga.pakkad || []).map(entry =>
        Array.isArray(entry) ? { note: entry[0], oct: entry[1] } : entry
    ).filter(e => SwaraMap[e.note] !== undefined && SwaraMap[e.note] !== null);

    const gapSteps = Math.max(1, Math.round(gapSec / stepSec));

    // Helper: schedule a single note at the current index position
    const t0 = audioCtx.currentTime + 0.05;
    let idx = 0;
    function schedNote(note, oct) {
        const ratio = SwaraMap[note];
        if (!ratio) return;
        let freq = baseSa * ratio;
        if (oct) freq *= Math.pow(2, oct);   // oct: -1=Mandra, 1=Taar, 2=Taar+1
        schedulePakkadNote(freq, inst, t0 + idx * stepSec, noteSec);
    }

    // Schedule Aroha
    if (arohaEntries.length) {
        arohaEntries.forEach(e => { schedNote(e.note, e.oct); idx++; });
        idx += gapSteps;
    }
    // Schedule Avaroha
    if (avarohaEntries.length) {
        avarohaEntries.forEach(e => { schedNote(e.note, e.oct); idx++; });
        idx += gapSteps;
    }
    // Schedule Pakkad
    if (pakkadEntries.length) {
        pakkadEntries.forEach(e => { schedNote(e.note, e.oct); idx++; });
    }

    if (idx === 0) {
        // Nothing to play
        if (btn) { btn.disabled = false; btn.textContent = '▶ Play'; btn.onclick = playPakkad; }
        return;
    }

    // Single JS timer for UI reset after all audio finishes
    const totalMs = (idx * stepSec + noteSec + 0.15) * 1000;
    pakkadEndTimer = setTimeout(() => {
        pakkadSources = [];
        if (btn) { btn.disabled = false; btn.textContent = '▶ Play'; btn.onclick = playPakkad; }
    }, totalMs);
}

function stopPakkad() {
    clearTimeout(pakkadEndTimer);
    pakkadEndTimer = null;
    for (const src of pakkadSources) {
        try { src.stop(); } catch(e) {}
    }
    pakkadSources = [];
    const btn = document.getElementById('raga-pakkad-play');
    if (btn) { btn.disabled = false; btn.textContent = '▶ Play'; btn.onclick = playPakkad; }
}

