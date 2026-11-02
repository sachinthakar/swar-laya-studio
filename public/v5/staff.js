// Swar-Laya Studio v5 — staff: read-only Western staff-notation popup
// (sibling to bhatkhande.js -- same data source, same read-only philosophy:
// renders whatever composition is currently in the editor via the shared
// buildCompositionData(), never re-parses the edit grid's DOM.)
//
// Pitch convention: Sa is drawn as middle C (treble clef), regardless of
// the actual tuning frequency chosen in Scale (Sa) -- that control sets
// playback pitch only. This "movable Sa = fixed Do" convention is the
// standard way Indian notation is rendered on a Western staff, so the
// shapes on the page stay identical no matter what key a performer sings
// or plays in.
//
// Swara -> staff letter + accidental (chromatic scale laid onto the 7
// diatonic letters, same choice most published Hindustani-notation-to-
// staff transcriptions use): natural letters for the shuddha swaras,
// flats for every komal swara, and a sharp only for tivra Ma (there is no
// "komal Sa" or "komal Pa", so those two never carry an accidental).
const SN_SWARA_PITCH = {
    'Sa': { letter: 'C', acc: 0 },
    're': { letter: 'D', acc: -1 },
    'Re': { letter: 'D', acc: 0 },
    'ga': { letter: 'E', acc: -1 },
    'Ga': { letter: 'E', acc: 0 },
    'Ma': { letter: 'F', acc: 0 },
    'ma': { letter: 'F', acc: 1 },
    'Pa': { letter: 'G', acc: 0 },
    'dha': { letter: 'A', acc: -1 },
    'Dha': { letter: 'A', acc: 0 },
    'ni': { letter: 'B', acc: -1 },
    'Ni': { letter: 'B', acc: 0 }
};
const SN_LETTER_OFFSET = { C: 0, D: 1, E: 2, F: 3, G: 4, A: 5, B: 6 };
const SN_ACC_GLYPH = { '-1': '♭', '0': '', '1': '♯' };  // flat / (none) / sharp

// A continuous "diatonic index" -- one unit per staff step (line-or-space),
//7 units per octave. Octave numbers follow the app's own scheme: the
// octave-button value (-2..2) stored per note is added straight to 4, so
// Sa at octave 0 always lands on middle C (index for C, octave 4).
function _snPitchIndex(letter, octave) { return octave * 7 + SN_LETTER_OFFSET[letter]; }

// Treble clef: bottom line E4, top line F5 -- 4 spaces, 8 diatonic steps.
const SN_STAFF_MIN = _snPitchIndex('E', 4);
const SN_STAFF_MAX = SN_STAFF_MIN + 8;
const SN_HALF_STEP = 6;     // px per diatonic step (so adjacent lines are 12px apart)
const SN_MARGIN     = 16;   // px of breathing room above the topmost element and below the bottommost

// Ledger lines needed between the staff and a note `d` diatonic steps
// beyond it (d = staffMin-idx below, or idx-staffMax above). Only every
// *other* step is a "line" position, same spacing as the staff itself.
function _snLedgerCount(d) { return Math.max(0, Math.floor(d / 2)); }

function _snEscapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

// Resolve one line's notes into a flat, time-ordered list of drawable
// events, carrying the previous pitch forward through '~' (sustain) the
// same way the audio engine does (sustain = keep sounding the last note,
// don't re-attack), and leaving rests ('-') with no pitch at all.
function _snResolveEvents(line) {
    const nBeats = (line.notes || []).length;
    const events = [];
    let lastIdx = null, lastAcc = 0, lastLetter = null;
    for (let b = 0; b < nBeats; b++) {
        const subs = line.notes[b] || ['-'];
        const octs = (line.octaves && line.octaves[b]) || [];
        const mee  = (line.meend  && line.meend[b])  || [];
        subs.forEach((sw, si) => {
            const ev = { beat: b, sub: si, nSub: subs.length, meend: !!mee[si] };
            if (sw === '-') {
                ev.rest = true;
            } else if (sw === '~') {
                ev.tie = true;
                ev.idx = lastIdx; ev.acc = lastAcc; ev.letter = lastLetter;
            } else {
                const p = SN_SWARA_PITCH[sw];
                if (!p) {
                    ev.unknown = true; ev.label = sw;
                } else {
                    const realOct = 4 + (octs[si] || 0);
                    ev.idx = _snPitchIndex(p.letter, realOct);
                    ev.acc = p.acc; ev.letter = p.letter;
                    lastIdx = ev.idx; lastAcc = ev.acc; lastLetter = ev.letter;
                }
            }
            events.push(ev);
        });
    }
    return events;
}

// One note's SVG markup (head + accidental + ledger lines + optional tie
// into the next note) at a given (x, y).
function _snNoteSvg(ev, x, y, nextX, nextY) {
    let out = '';
    const ledgerXr = 9;
    // Ledger lines, drawn outward from the staff toward this note.
    if (ev.idx < SN_STAFF_MIN) {
        const n = _snLedgerCount(SN_STAFF_MIN - ev.idx);
        for (let i = 0; i < n; i++) {
            const lIdx = SN_STAFF_MIN - 2 * (i + 1);
            const ly = y - (ev.idx - lIdx) * SN_HALF_STEP;
            out += `<line class="sn-ledger" x1="${x - ledgerXr}" x2="${x + ledgerXr}" y1="${ly}" y2="${ly}"/>`;
        }
    } else if (ev.idx > SN_STAFF_MAX) {
        const n = _snLedgerCount(ev.idx - SN_STAFF_MAX);
        for (let i = 0; i < n; i++) {
            const lIdx = SN_STAFF_MAX + 2 * (i + 1);
            const ly = y + (lIdx - ev.idx) * SN_HALF_STEP;
            out += `<line class="sn-ledger" x1="${x - ledgerXr}" x2="${x + ledgerXr}" y1="${ly}" y2="${ly}"/>`;
        }
    }
    // Meend-glide arc into the next note (drawn before the head so the
    // head sits on top of the arc's end point). A plain sustain ('~') is
    // shown only as a hollow notehead at the carried-forward pitch, same
    // as Bhatkhande View shows it as a bare symbol -- it does not itself
    // draw a connecting line, since what it continues may sit across an
    // intervening rest and a line there would misleadingly suggest a
    // glide to whatever pitch happens to follow.
    if (ev.meend && nextX != null) {
        const midX = (x + nextX) / 2;
        const arcH = 8;
        const topY = Math.min(y, nextY) - arcH;
        out += `<path class="sn-tie" d="M${x + 6},${y - 3} Q${midX},${topY} ${nextX - 6},${nextY - 3}"/>`;
    }
    const acc = SN_ACC_GLYPH[String(ev.acc || 0)];
    if (acc) out += `<text class="sn-acc" x="${x - 11}" y="${y + 4}">${acc}</text>`;
    out += `<ellipse class="sn-head${ev.tie ? ' sn-head-ghost' : ''}" cx="${x}" cy="${y}" rx="5" ry="4"/>`;
    return out;
}

// One taal-cycle line, rendered as one staff with a bol/Sam-Khali header
// row above it (mirrors _bkRenderLine's header) and a lyric row below.
function _snRenderLine(line, idx) {
    const taalKey = line.taal_key;
    const khalis  = (window.TaalKhali && TaalKhali[taalKey]) || [];
    const bols    = (line.percussion === 'dholak'
                        ? (window.DholakBols && DholakBols[taalKey])
                        : (window.TaalBols && TaalBols[taalKey])) || [];
    const lyrics  = line.lyrics || [];
    const nBeats  = (line.notes || []).length;
    const events  = _snResolveEvents(line);

    // Horizontal layout: fixed slot per beat, sub-notes divide it evenly.
    const BEAT_W = 70, LEFT_PAD = 56, RIGHT_PAD = 24, TOP_PAD = 34, BOTTOM_PAD = 26;
    const width  = LEFT_PAD + nBeats * BEAT_W + RIGHT_PAD;
    events.forEach(ev => {
        ev.x = LEFT_PAD + ev.beat * BEAT_W + BEAT_W * (ev.sub + 0.5) / ev.nSub;
    });

    // Vertical range this line actually needs (always at least the 5
    // staff lines themselves), so a simple melody gets a compact staff
    // and a wide-ranging one still fits without clipping.
    const pitched = events.filter(e => e.idx != null);
    const minIdx = Math.min(SN_STAFF_MIN, ...(pitched.length ? pitched.map(e => e.idx) : [SN_STAFF_MIN]));
    const maxIdx = Math.max(SN_STAFF_MAX, ...(pitched.length ? pitched.map(e => e.idx) : [SN_STAFF_MAX]));
    const topIdx = maxIdx + 2, botIdx = minIdx - 2;   // 1 extra step of breathing room
    const yOf = i => TOP_PAD + (topIdx - i) * SN_HALF_STEP;
    const height = TOP_PAD + (topIdx - botIdx) * SN_HALF_STEP + BOTTOM_PAD;

    events.forEach(ev => { if (ev.idx != null) ev.y = yOf(ev.idx); });

    // Staff lines + clef + opening/closing barlines.
    let svg = '';
    for (let i = 0; i <= 4; i++) {
        const lineIdx = SN_STAFF_MIN + i * 2;
        const ly = yOf(lineIdx);
        svg += `<line class="sn-staffline" x1="${LEFT_PAD - 10}" x2="${width - RIGHT_PAD + 6}" y1="${ly}" y2="${ly}"/>`;
    }
    const staffTopY = yOf(SN_STAFF_MAX), staffBotY = yOf(SN_STAFF_MIN);
    svg += `<text class="sn-clef" x="${LEFT_PAD - 38}" y="${staffBotY - 2}">&#119070;</text>`;
    svg += `<line class="sn-barline" x1="${LEFT_PAD - 10}" x2="${LEFT_PAD - 10}" y1="${staffTopY}" y2="${staffBotY}"/>`;
    svg += `<line class="sn-barline" x1="${width - RIGHT_PAD + 6}" x2="${width - RIGHT_PAD + 6}" y1="${staffTopY}" y2="${staffBotY}"/>`;

    // Beat bols / Sam ('X') / Khali ('0') header, and a thin tick + optional
    // lyric syllable under each beat's first sub-note.
    for (let b = 0; b < nBeats; b++) {
        const bx      = LEFT_PAD + b * BEAT_W + BEAT_W / 2;
        const isSam   = b === 0;
        const isKhali = !isSam && khalis.includes(b);
        const marker  = isSam ? 'X' : (isKhali ? '0' : '');
        svg += `<text class="sn-bol${isKhali ? ' sn-khali' : ''}" x="${bx}" y="14">${_snEscapeHtml(bols[b] || '')}</text>`;
        svg += `<text class="sn-marker${isSam ? ' sn-sam' : ''}${isKhali ? ' sn-khali' : ''}" x="${bx}" y="27">${marker}${marker ? '' : b + 1}</text>`;
        if (lyrics[b]) svg += `<text class="sn-lyric" x="${bx}" y="${height - 8}">${_snEscapeHtml(lyrics[b])}</text>`;
        if (b > 0) svg += `<line class="sn-beat-tick" x1="${LEFT_PAD + b * BEAT_W}" x2="${LEFT_PAD + b * BEAT_W}" y1="${staffTopY - 4}" y2="${staffBotY + 4}"/>`;
    }

    // Notes last, so ties/ledgers sit correctly relative to the grid above.
    events.forEach((ev, i) => {
        if (ev.rest) {
            svg += `<text class="sn-rest" x="${ev.x}" y="${(staffTopY + staffBotY) / 2 + 5}">&#119101;</text>`;
            return;
        }
        if (ev.unknown) {
            svg += `<text class="sn-unknown" x="${ev.x}" y="${(staffTopY + staffBotY) / 2 + 5}" title="Unrecognised note">?</text>`;
            return;
        }
        const next = events[i + 1];
        svg += _snNoteSvg(ev, ev.x, ev.y, next && !next.rest ? next.x : null, next ? next.y : null);
    });

    const tags = [];
    if (line.section) tags.push(_snEscapeHtml(line.section));
    if (line.adlib) tags.push('<span class="bk-adlib-tag" title="Free-tempo adlib -- still shown on the fixed beat grid stored for this line">adlib</span>');
    if (line.line_bpm) tags.push(`${line.line_bpm} BPM`);

    return `<div class="sn-cycle">
        <div class="bk-cycle-header">Line ${idx + 1}
            <span class="bk-section-tag">${_bkTaalLabel(taalKey)}, ${nBeats} beats${tags.length ? ' &middot; ' + tags.join(' &middot; ') : ''}</span>
        </div>
        <svg class="sn-svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">${svg}</svg>
    </div>`;
}

function _snBuildHtml(data) {
    const lines = (data.lines || []).map((l, i) => _snRenderLine(l, i)).join('');
    const anyAdlib = (data.lines || []).some(l => l.adlib);
    return `
        <div class="bk-page-title">${_bkEscapeHtml(data.title || 'Untitled')}</div>
        <div class="bk-page-sub">Sa is written as middle C (treble clef) &nbsp;&middot;&nbsp; actual tonic (Sa): ${_bkScaleLabel(data.scale)} &nbsp;&middot;&nbsp; Tempo: ${data.bpm} BPM</div>
        ${anyAdlib ? '<div class="bk-adlib-note">One or more lines are flagged adlib (free tempo) but still shown on the fixed beat grid stored for them -- this view is only as free-timed as the saved data.</div>' : ''}
        ${lines || '<div class="bk-empty">Nothing to show yet -- add a line to the song sheet first.</div>'}
        <div class="bk-legend">
            <b>Legend:</b> X = Sam &middot; 0 = Khali &middot; &#9837; = komal (flat) &middot; &#9839; = tivra Ma (sharp) &middot;
            hollow note = sustain (continues the previous pitch) &middot; curved line = meend (glide) into the next note &middot;
            &#119101; = rest &middot; notes beyond the 5 lines use short ledger lines, same as any Western staff.
        </div>`;
}

function openStaffView() {
    const data = buildCompositionData();
    document.getElementById('staff-content').innerHTML = _snBuildHtml(data);
    document.getElementById('staff-modal').style.display = 'flex';
}

function closeStaffView() {
    document.getElementById('staff-modal').style.display = 'none';
}

function printStaffView() { _printModal('staff-modal'); }
