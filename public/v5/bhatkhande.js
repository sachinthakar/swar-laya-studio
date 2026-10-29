// Swar-Laya Studio v5 — bhatkhande: read-only traditional-notation popup
// (split from the single-file studio.html; scripts load in order and share globals)
//
// Renders whatever composition is currently in the editor (via the same
// buildCompositionData() the Save/Export buttons use -- never re-parses the
// edit grid's DOM or any rendered HTML, same lesson as the Python pipeline's
// replay script: always read the canonical data, not a display of it) as a
// read-only table in the traditional book/teacher-notes layout: one row per
// taal cycle, beat numbers with Sam ('X') / Khali ('0') marks (reusing the
// app's own TaalBeats/TaalKhali/TaalBols -- no vibhaag/tali structure is
// invented here beyond what those tables already encode, since e.g.
// TaalKhali for "16" (Teentaal) only marks beat 9 as Khali and does not
// track the 2 intermediate Tali positions a full traditional printed score
// would show; this view marks exactly what the app's data model tracks,
// nothing more).
//
// Octave dots: this app supports FIVE octave levels (-2,-1,0,1,2 via the
// octave-btn data-oct values in sheet.js's _makeNoteUnit), not just the
// simple upper/lower the Python matrix generator handled -- one dot per
// octave step, above the swara for Taar (+1/+2), below for Mandra (-1/-2).
//
// Sub-notes (gamak/taan, e.g. Sa-Re-Ga-Ma in one beat) render as "a / b / c"
// in time order, same convention as the Python-generated matrices.
//
// Known note-vocabulary mismatch worth being aware of: this app's SwaraMap
// uses lowercase "ma" for tivra Ma (consistent with its own lower=altered /
// upper=natural case convention for every other swara); the separate Python
// transcription pipeline (transcribe_song_studio_v2.py) instead emits
// "Ma'" for the same swara. A JSON made by the Python pipeline and imported
// here will show a blank/unrecognized note-select for any tivra-Ma beat
// until that's reconciled -- out of scope for this popup, flagging it here
// so it isn't mistaken for a bug in this file.

function _bkEscapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

// Human label for a taal_key, read from the existing #taal-select options
// (single source of truth -- no parallel label map to drift out of sync).
function _bkTaalLabel(taalKey) {
    const opt = document.querySelector(`#taal-select option[value="${CSS.escape(String(taalKey))}"]`);
    return opt ? opt.textContent : (taalKey || '?');
}

// Human label for a scale (tonic) value, read from #scale-select the same way.
function _bkScaleLabel(scaleVal) {
    const v = parseFloat(scaleVal);
    let best = null, bestDiff = Infinity;
    document.querySelectorAll('#scale-select option').forEach(opt => {
        const diff = Math.abs(parseFloat(opt.value) - v);
        if (diff < bestDiff) { bestDiff = diff; best = opt; }
    });
    return best ? best.textContent : `${v.toFixed(2)} Hz`;
}

// One dot per octave step, above the swara for +1/+2 (Taar), below for -1/-2
// (Mandra) -- matches the octave-btn convention in sheet.js exactly (that's
// the one place this count is defined; read it from there, don't guess).
function _bkOctaveDotsHtml(ov) {
    const n = Math.abs(ov || 0);
    if (n === 0) return '';
    return '<span class="bk-dot"></span>'.repeat(n);
}

function _bkSwaraHtml(sw, ov) {
    if (sw === '-') return '<span class="bk-rest">&mdash;</span>';
    if (sw === '~') return '<span class="bk-sustain">~</span>';
    const dots = _bkOctaveDotsHtml(ov);
    const above = (ov > 0) ? `<span class="bk-oct-row">${dots}</span>` : '';
    const below = (ov < 0) ? `<span class="bk-oct-row">${dots}</span>` : '';
    return `<span class="bk-swara">${above}<span class="bk-swara-text">${_bkEscapeHtml(sw)}</span>${below}</span>`;
}

// One beat cell: 1-4 sub-notes (gamak/taan), joined "a / b / c" in time order.
// meendArr[i] true marks that sub-note as gliding into the next -- shown with
// a small tie mark, same meaning as the Python matrix's legend entry.
function _bkBeatCellHtml(notesArr, octsArr, meendArr) {
    const parts = (notesArr || ['-']).map((sw, i) => {
        let html = _bkSwaraHtml(sw, (octsArr && octsArr[i]) || 0);
        if (meendArr && meendArr[i]) {
            html += '<span class="bk-meend" title="Meend (glide) into the next note">&#8781;</span>';
        }
        return html;
    });
    return parts.join('<span class="bk-subsplit">/</span>');
}

// One taal-cycle row for one line: beat-number header (with Sam 'X' / Khali
// '0' from TaalBeats/TaalKhali) above a single row of beat cells. Uses the
// line's own notes.length as the beat count (not TaalBeats[taalKey]) so an
// adlib line with a user-adjusted (+Beat/-Beat) length renders correctly
// rather than being forced back to the taal's "natural" cycle length.
function _bkRenderLine(line, idx) {
    const taalKey = line.taal_key;
    const khalis  = (window.TaalKhali && TaalKhali[taalKey]) || [];
    const bols    = (line.percussion === 'dholak'
                        ? (window.DholakBols && DholakBols[taalKey])
                        : (window.TaalBols && TaalBols[taalKey])) || [];
    const nBeats  = (line.notes || []).length;

    let headCells = '';
    let bodyCells = '';
    for (let i = 0; i < nBeats; i++) {
        const isSam   = i === 0;
        const isKhali = !isSam && khalis.includes(i);
        const marker  = isSam ? 'X' : (isKhali ? '0' : '');
        headCells += `<th class="bk-beat-head${isSam ? ' bk-sam' : ''}${isKhali ? ' bk-khali' : ''}">` +
                     `<div class="bk-bol">${_bkEscapeHtml(bols[i] || '')}</div>` +
                     `<div class="bk-marker">${marker}</div>${i + 1}</th>`;
        bodyCells += `<td class="bk-swara-cell">${_bkBeatCellHtml(line.notes[i], line.octaves[i], line.meend && line.meend[i])}</td>`;
    }

    const tags = [];
    if (line.section) tags.push(_bkEscapeHtml(line.section));
    if (line.adlib) tags.push('<span class="bk-adlib-tag" title="Free-tempo adlib -- still shown on the fixed beat grid stored for this line, not true free timing">adlib</span>');
    if (line.line_bpm) tags.push(`${line.line_bpm} BPM`);

    return `<div class="bk-cycle">
        <div class="bk-cycle-header">Line ${idx + 1}
            <span class="bk-section-tag">${_bkTaalLabel(taalKey)}, ${nBeats} beats${tags.length ? ' &middot; ' + tags.join(' &middot; ') : ''}</span>
        </div>
        <table class="bk-matrix"><thead><tr>${headCells}</tr></thead>
        <tbody><tr>${bodyCells}</tr></tbody></table>
    </div>`;
}

function _bkBuildHtml(data) {
    const lines = (data.lines || []).map((l, i) => _bkRenderLine(l, i)).join('');
    const anyAdlib = (data.lines || []).some(l => l.adlib);
    return `
        <div class="bk-page-title">${_bkEscapeHtml(data.title || 'Untitled')}</div>
        <div class="bk-page-sub">Tonic (Sa): ${_bkScaleLabel(data.scale)} &nbsp;&middot;&nbsp; Tempo: ${data.bpm} BPM</div>
        ${anyAdlib ? '<div class="bk-adlib-note">One or more lines are flagged adlib (free tempo) but still shown on the fixed beat grid stored for them -- this view is only as free-timed as the saved data.</div>' : ''}
        ${lines || '<div class="bk-empty">Nothing to show yet -- add a line to the song sheet first.</div>'}
        <div class="bk-legend">
            <b>Legend:</b> X = Sam &middot; 0 = Khali &middot; a / b = sub-notes (gamak/taan) in time order &middot;
            dot above/below = octave (Taar/Mandra, one dot per step) &middot; ~ = sustain &middot; &mdash; = rest &middot;
            &#8781; = meend (glide) into the next note
        </div>`;
}

function openBhatkhandeView() {
    const data = buildCompositionData();
    document.getElementById('bhatkhande-content').innerHTML = _bkBuildHtml(data);
    document.getElementById('bhatkhande-modal').style.display = 'flex';
}

function closeBhatkhandeView() {
    document.getElementById('bhatkhande-modal').style.display = 'none';
}
