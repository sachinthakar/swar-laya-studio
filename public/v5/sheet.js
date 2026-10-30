// Swar-Laya Studio v5 — song sheet: note cells, line builder, taal ruler, settings, save/load JSON, startup
// (split from the single-file studio.html; scripts load in order and share globals)

// Apply / remove the visual "hold" style on a note-unit based on its current value.
// Called after creating each unit (on load) and on every user change.
function applySustainStyle(unit) {
    const sel  = unit.querySelector('.note-select');
    const hold = sel && sel.value === '~';
    // Octave buttons are meaningless on a hold beat — keep them disabled.
    // Meend IS meaningful on a hold: it marks the sustained period as a slide
    // toward the next sounding note, so it stays clickable.
    unit.querySelectorAll('.octave-btn').forEach(btn => {
        btn.disabled      = hold;
        btn.style.opacity = hold ? '0.25' : '';
    });
    const meendBtn = unit.querySelector('.meend-btn');
    if (meendBtn) {
        meendBtn.disabled      = false;
        meendBtn.style.opacity = '';
    }
    if (sel) {
        sel.style.backgroundColor = hold ? '#e3f2fd' : '';
        sel.style.color           = hold ? '#1565c0' : '';
        sel.style.fontStyle       = hold ? 'italic'  : '';
    }
}

// Count consecutive "~" beats immediately after fromBeatIdx within the same line.
// Used by the playback engine to extend the note duration across held beats.
function countSustains(targetLine, fromBeatIdx) {
    const notesRow = targetLine.querySelector('.grid-row[data-row="notes"]');
    if (!notesRow) return 0;
    const cells = notesRow.querySelectorAll('.cell');
    let n = 0;
    for (let i = fromBeatIdx + 1; i < cells.length; i++) {
        const sel = cells[i].querySelector('.note-select');
        if (sel && sel.value === '~') n++;
        else break;
    }
    return n;
}

// ─── Sub-note helpers ─────────────────────────────────────────────────────────

/**
 * Build one note-unit <div> (a single sub-beat slot inside a beat cell).
 * showRemove: show the "×" remove button (true when the beat has ≥2 sub-notes).
 */
function _makeNoteUnit(nv, ov, mv, allOpts, showRemove) {
    // Replace the pre-built allOpts with a version that selects the right note
    const opts = allOpts.replace(
        `<option>${nv}</option>`,
        `<option selected>${nv}</option>`
    );
    const unit = document.createElement('div');
    unit.className = 'note-unit';
    unit.innerHTML =
        `<select class="note-select">${opts}</select>` +
        `<span class="octave-controls">` +
        `<button class="octave-btn${ov===2?' active':''}" data-oct="2"  onclick="toggleOctave(this,2)"  title="Taar+1 (2nd upper octave)">&#730;&#730;</button>` +
        `<button class="octave-btn${ov===1?' active':''}" data-oct="1"  onclick="toggleOctave(this,1)"  title="Taar (upper octave)">&#730;</button>` +
        `<button class="octave-btn${ov===-1?' active':''}" data-oct="-1" onclick="toggleOctave(this,-1)" title="Mandra (lower octave)">&#x2022;</button>` +
        `<button class="octave-btn${ov===-2?' active':''}" data-oct="-2" onclick="toggleOctave(this,-2)" title="Mandra-2 (2nd lower octave)">&#x2022;&#x2022;</button>` +
        `</span>` +
        `<button class="meend-btn${mv?' active':''}" onclick="this.classList.toggle('active')" title="Meend (glide to this note)">m</button>` +
        `<button class="rem-sub-btn" onclick="removeSubNote(this)" title="Remove sub-note" ` +
        `style="visibility:${showRemove?'visible':'hidden'}">&#x2715;</button>`;
    applySustainStyle(unit);
    return unit;
}

/**
 * Remove a sub-note unit.  Only removes when ≥2 units exist; hides the
 * remove button on the last remaining unit.
 */
function removeSubNote(btn) {
    const unit = btn.closest('.note-unit');
    const sub  = unit.parentElement;   // .sub-cell-container
    const all  = sub.querySelectorAll('.note-unit');
    if (all.length <= 1) return;      // never remove the last note-unit
    sub.removeChild(unit);
    // If only 1 left, hide its remove button
    if (sub.querySelectorAll('.note-unit').length === 1) {
        const rem = sub.querySelector('.rem-sub-btn');
        if (rem) rem.style.visibility = 'hidden';
    }
    applyRagaHighlights();
}

// ─── Line Builder ─────────────────────────────────────────────────────────────

/** Re-label every line badge: "Line 1", "Line 2", … */
function renumberLines() {
    document.querySelectorAll('.line-badge').forEach((b, i) => {
        b.textContent = `Line ${i + 1}`;
    });
}

/**
 * addLine(data, insertBeforeEl)
 *   data           — JSON line object (null = blank line using toolbar taal)
 *   insertBeforeEl — optional .line-container to insert before (null = append)
 */
function addLine(data, insertBeforeEl) {
    const taalKey = (data && data.taal_key) ? data.taal_key : document.getElementById('taal-select').value;
    if (!taalKey) {
        alert('Please choose a Taal first using the Taal dropdown in the toolbar.');
        return;
    }
    // Adlib lines loaded from JSON may have more beats than the taal — honour that.
    const beats = (data && data.adlib && data.beats)
                  ? parseInt(data.beats)
                  : (TaalBeats[taalKey] || parseInt(taalKey) || 8);

    // Per-beat sub-note counts are handled individually in the notes-row loop below.
    // (Global maxSubs removed: each beat now shows only as many columns as it has notes.)

    const instVal    = data ? (data.line_instrument || 'harmonium') : 'harmonium';
    // New JSON uses percussion:'tabla'|'dholak'|'mute'; old JSON used tabla_mute:bool
    const percVal    = data ? (data.percussion || (data.tabla_mute ? 'mute' : 'tabla')) : 'tabla';
    const volVal     = data ? (data.line_volume !== undefined ? data.line_volume : 1.0) : 1.0;
    const section    = data ? (data.section || '') : '';
    const adlibVal   = data ? !!data.adlib        : false;
    const legatoVal  = data ? !!data.legato       : false;
    const lineBpmVal = data ? (data.line_bpm > 0 ? data.line_bpm : '') : '';

    const sheet = document.getElementById('song-sheet');
    const line  = document.createElement('div');
    line.className = 'line-container';
    line.setAttribute('data-beats', beats);
    line.setAttribute('data-taal',  taalKey);

    // Badge
    const badge = document.createElement('span');
    badge.className = 'line-badge';
    badge.textContent = `Line ${sheet.children.length + 1}`;
    line.appendChild(badge);

    // Settings bar
    const settings = document.createElement('div');
    settings.className = 'line-settings';
    settings.innerHTML = `
        <label>Instrument:</label>
        <select class="line-inst-select" style="width:108px;">
            <option value="harmonium" ${instVal==='harmonium'?'selected':''}>Harmonium</option>
            <option value="flute"     ${instVal==='flute'    ?'selected':''}>Flute</option>
            <option value="sitar"     ${instVal==='sitar'    ?'selected':''}>Sitar</option>
            <option value="violin"    ${instVal==='violin'   ?'selected':''}>Violin</option>
            <option value="piano"     ${instVal==='piano'    ?'selected':''}>Piano</option>
            <option value="bellpad"   ${instVal==='bellpad'  ?'selected':''}>Bellpad</option>
        </select>
        <label>Percussion:</label>
        <select class="line-perc-sel" style="width:82px;">
            <option value="tabla"  ${percVal==='tabla' ?'selected':''}>Tabla</option>
            <option value="dholak" ${percVal==='dholak'?'selected':''}>Dholak</option>
            <option value="mute"   ${percVal==='mute'  ?'selected':''}>Mute</option>
        </select>
        <label>Volume:</label>
        <input type="range" class="line-vol" min="0.1" max="1" step="0.05" value="${volVal}" style="width:70px;">
        <label>Section:</label>
        <input type="text" class="line-section" value="${section}" style="width:80px;" placeholder="e.g. Mukhda">
        <label title="Legato: notes snap on immediately and sustain until the next note, so the melody flows without gaps between beats.">Legato:</label>
        <input type="checkbox" class="line-legato" ${legatoVal ? 'checked' : ''}
               title="Legato: 8 ms attack/release so consecutive notes connect smoothly."
               style="width:16px;">
        <label class="adlib-label">Adlib/Alap:</label>
        <input type="checkbox" class="line-adlib" ${adlibVal ? 'checked' : ''}
               title="Adlib/Alap mode: free tempo, tabla silenced. Use Line BPM to set the alap speed."
               style="width:16px;">
        <label>Line BPM:</label>
        <input type="number" class="line-bpm" min="10" max="300" step="any" value="${lineBpmVal}"
               placeholder="(master)" style="width:62px;"
               title="Per-line tempo override. Leave blank to use the master BPM. Set a slow value (e.g. 30–50) for Alap lines.">
    `;
    line.appendChild(settings);

    // Adlib beat-controls strip — shown only when adlib is ON
    const beatStrip = document.createElement('div');
    beatStrip.className = 'adlib-beat-controls';
    beatStrip.style.display = adlibVal ? 'flex' : 'none';
    beatStrip.innerHTML = `
        <span class="adlib-beat-controls-label">&#127925; Adlib beats:</span>
        <button class="btn-rem-beat" title="Remove last beat column (min 1)">&#8722; Beat</button>
        <span class="adlib-beat-count">${beats} beats</span>
        <button class="btn-add-beat" title="Add a beat column to this adlib line">+ Beat</button>
        <span style="font-size:11px; color:#888; margin-left:8px;">Use Line BPM to control note speed</span>
    `;
    line.appendChild(beatStrip);

    // Lyrics row
    const rowL = document.createElement('div');
    rowL.className = 'grid-row';
    rowL.setAttribute('data-row', 'lyrics');
    rowL.innerHTML = `<div class="row-label">Lyrics</div>`;
    for (let i = 0; i < beats; i++) {
        const lv = data && data.lyrics ? (data.lyrics[i] || '') : '';
        rowL.innerHTML += `<div class="cell"><input type="text" class="lyr-input lyr" value="${lv}"></div>`;
    }

    // Notes row — each beat gets exactly as many note-unit rows as it has sub-notes.
    // Gamak beats (e.g. GaPaGaDha in one beat) have multiple stacked note-units;
    // single-note beats have just one.  The "+" button adds a sub-note for gamak
    // entry; "×" on each unit removes it (hidden when only 1 sub-note remains).
    const rowN = document.createElement('div');
    rowN.className = 'grid-row';
    rowN.setAttribute('data-row', 'notes');
    rowN.innerHTML = `<div class="row-label">Notes</div>`;

    // Pre-build option string once (shared by all selects)
    let _allOpts = '';
    for (const sw in SwaraMap) _allOpts += `<option>${sw}</option>`;

    for (let i = 0; i < beats; i++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.dataset.beatIdx = i;
        const sub = document.createElement('div');
        sub.className = 'sub-cell-container';

        // How many sub-notes for THIS beat (default 1)
        const beatSubs = (data && data.notes && data.notes[i] && data.notes[i].length > 0)
                         ? data.notes[i].length : 1;

        for (let s = 0; s < beatSubs; s++) {
            const nv = data && data.notes   && data.notes[i]   ? (data.notes[i][s]   ?? '-')    : '-';
            const ov = data && data.octaves && data.octaves[i] ? (data.octaves[i][s] ??  0 )    :  0;
            const mv = data && data.meend   && data.meend[i]   ? (data.meend[i][s]   ?? false)  : false;
            sub.appendChild(_makeNoteUnit(nv, ov, mv, _allOpts, beatSubs > 1));
        }

        // "+" add-sub-note button (always present, small)
        const addBtn = document.createElement('button');
        addBtn.className = 'add-sub-btn';
        addBtn.title = 'Add sub-note (gamak / taan)';
        addBtn.textContent = '+';
        addBtn.addEventListener('click', () => {
            sub.appendChild(_makeNoteUnit('-', 0, false, _allOpts, true));
            // Show remove buttons on all units now that there are 2+
            sub.querySelectorAll('.rem-sub-btn').forEach(b => b.style.visibility = '');
        });

        cell.appendChild(sub);
        cell.appendChild(addBtn);
        rowN.appendChild(cell);
    }

    line.appendChild(rowL);
    line.appendChild(rowN);

    // "Insert Before" button (top-right, just left of the remove button)
    const insBtn = document.createElement('button');
    // ▶ Play this line only (loops until Stop is pressed)
    const playBtn = document.createElement('button');
    playBtn.className   = 'btn-play-line';
    playBtn.title       = 'Play this line only (once, or loops if Loop is checked)';
    playBtn.textContent = '▶ Line';
    playBtn.addEventListener('click', () => {
        if (playBtn.classList.contains('active')) {
            stopPlayback();   // acts as toggle-stop
        } else {
            startPlayback(line);   // stopPlayback() is called inside; it clears all active buttons first
            playBtn.classList.add('active');
        }
    });
    line.appendChild(playBtn);

    insBtn.className   = 'btn-insert-before';
    insBtn.title       = 'Insert a blank line before this one';
    insBtn.textContent = '⤴ Insert Before';
    insBtn.addEventListener('click', () => addLine(null, line));
    line.appendChild(insBtn);

    const rem = document.createElement('button');
    rem.className   = 'btn-remove';
    rem.textContent = '×';
    rem.onclick = () => { line.remove(); renumberLines(); };
    line.appendChild(rem);

    if (adlibVal) line.classList.add('adlib-line');
    // Audio-transcription metadata (seconds in the source recording)
    if (data && data.audio_start != null && data.audio_end != null) {
        line.dataset.audioStart = data.audio_start;
        line.dataset.audioEnd   = data.audio_end;
    }
    // Extension hook (transcribe.js adds per-line audio controls)
    if (typeof onLineCreated === 'function') onLineCreated(line, data);
    if (insertBeforeEl) {
        sheet.insertBefore(line, insertBeforeEl);
    } else {
        sheet.appendChild(line);
    }
    renumberLines();
    applyRagaHighlights();
    updateScrollMargin();
    return line;
}

function toggleOctave(btn, val) {
    const parent = btn.closest('.octave-controls');
    const wasActive = btn.classList.contains('active');
    parent.querySelectorAll('.octave-btn').forEach(b => b.classList.remove('active'));
    if (!wasActive) btn.classList.add('active');
}

// ─── Settings Persistence (localStorage) ─────────────────────────────────────

/** Save taal, BPM and scale to localStorage so they survive a page reload. */
function saveSettings() {
    localStorage.setItem('sl_taal',       document.getElementById('taal-select').value);
    localStorage.setItem('sl_bpm',        document.getElementById('bpm').value);
    localStorage.setItem('sl_scale',      document.getElementById('scale-select').value);
    localStorage.setItem('sl_master_vol', document.getElementById('master-vol').value);
}

/**
 * Restore taal, BPM and scale from localStorage.
 * Called once at startup, before the initial addLine().
 */
function restoreSettings() {
    const taal      = localStorage.getItem('sl_taal');
    const bpm       = localStorage.getItem('sl_bpm');
    const scale     = localStorage.getItem('sl_scale');
    const masterVol = localStorage.getItem('sl_master_vol');
    if (taal)  document.getElementById('taal-select').value  = taal;
    if (bpm)   document.getElementById('bpm').value          = bpm;
    if (scale) {
        // Migrate old format: scale was stored as half the Sa Hz (pre-fix).
        // New format stores the actual Sa Hz. Threshold 256 safely separates
        // old max (246.94) from new min (261.63).
        const sv = parseFloat(scale) < 256 ? parseFloat(scale) * 2 : parseFloat(scale);
        document.getElementById('scale-select').value = sv;
    }
    if (masterVol) {
        document.getElementById('master-vol').value = masterVol;
        const display = document.getElementById('master-vol-display');
        if (display) display.textContent = parseFloat(masterVol).toFixed(1) + '×';
    }
    // Render ruler for restored taal (if any)
    if (taal) renderTaalRuler(taal, 'tabla');
}

// ─── Taal Ruler ───────────────────────────────────────────────────────────────

/**
 * Render (or update) the shared sticky taal ruler above the song sheet.
 * Called on taal change, on restore, and each time playback enters a new line.
 */
function renderTaalRuler(taalKey, percType) {
    const ruler = document.getElementById('taal-ruler');
    if (!taalKey) { ruler.className = ''; ruler.innerHTML = ''; updateScrollMargin(); return; }

    const beats  = TaalBeats[taalKey] || parseInt(taalKey) || 0;
    const bols   = ((percType === 'dholak') ? DholakBols[taalKey] : TaalBols[taalKey]) || [];
    const khalis = TaalKhali[taalKey] || [];

    let html = '<div class="row-label">Beat</div>';
    for (let i = 0; i < beats; i++) {
        const isSam   = i === 0;
        const isKhali = !isSam && khalis.includes(i);
        const marker  = isSam ? 'X' : (isKhali ? '0' : '');
        html +=
            `<div class="cell beat-num${isSam?' sam':''}${isKhali?' khali':''}" data-beat-idx="${i}">` +
            `<div class="bol-text">${bols[i] || ''}</div>` +
            `<div class="taal-marker">${marker}</div>${i + 1}</div>`;
    }
    ruler.innerHTML = html;
    ruler.className = 'visible';
    updateScrollMargin();
}

/** Keep .line-container scroll-margin-top in sync with the sticky header height. */
function updateScrollMargin() {
    const h = (document.getElementById('sticky-header')?.offsetHeight || 80) + 12;
    document.querySelectorAll('.line-container').forEach(el => {
        el.style.scrollMarginTop = h + 'px';
    });
}

// ─── Full Screen ──────────────────────────────────────────────────────────────

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => {
            alert(`Fullscreen not available: ${err.message}`);
        });
    } else {
        document.exitFullscreen();
    }
}

document.addEventListener('fullscreenchange', () => {
    const btn = document.getElementById('btn-fullscreen');
    if (btn) btn.textContent = document.fullscreenElement ? '⊡ Exit Full Screen' : '⛶ Full Screen';
});

// ─── Save / Load ──────────────────────────────────────────────────────────────

// Shared helper: collect the current sheet into the JSON data object.
// Used by both local export (downloadFile) and cloud save (saveToCloud).
// One .line-container → plain JSON line object (same schema as file export)
function lineToData(l) {
        const lyrics = Array.from(l.querySelectorAll('.lyr')).map(el => el.value);
        const subs   = l.querySelectorAll('.sub-cell-container');
        const notes  = [], octaves = [], meend = [];

        subs.forEach(sub => {
            const units = sub.querySelectorAll('.note-unit');
            const nRow  = [], oRow = [], mRow = [];
            units.forEach(unit => {
                nRow.push(unit.querySelector('.note-select').value);
                const ao = unit.querySelector('.octave-btn.active');
                oRow.push(ao ? parseInt(ao.dataset.oct) : 0);
                mRow.push(unit.querySelector('.meend-btn').classList.contains('active'));
            });
            notes.push(nRow); octaves.push(oRow); meend.push(mRow);
        });

        const lineBpmSaved = parseFloat(l.querySelector('.line-bpm').value) || null;
        const out = {
            section:         l.querySelector('.line-section').value,
            line_instrument: l.querySelector('.line-inst-select').value,
            percussion:      l.querySelector('.line-perc-sel')?.value || 'tabla',
            line_volume:     parseFloat(l.querySelector('.line-vol').value),
            adlib:           l.querySelector('.line-adlib').checked,
            legato:          l.querySelector('.line-legato')?.checked || false,
            line_bpm:        lineBpmSaved,
            taal_key:        l.getAttribute('data-taal'),
            beats:           l.getAttribute('data-beats'),
            lyrics, notes, octaves, meend
        };
        if (l.dataset.audioStart !== undefined) {
            out.audio_start = parseFloat(l.dataset.audioStart);
            out.audio_end   = parseFloat(l.dataset.audioEnd);
        }
        return out;
}

function buildCompositionData() {
    const lineArray = Array.from(document.querySelectorAll('.line-container')).map(lineToData);
    return {
        title:    document.getElementById('song-title').value || 'Untitled',
        scale:    document.getElementById('scale-select').value,
        bpm:      parseInt(document.getElementById('bpm').value) || 90,
        pre_roll: parseInt(document.getElementById('pre-roll-select').value) || 0,
        lines:    lineArray
    };
}

// Export to local JSON file (no login required — always available as backup)
function downloadFile() {
    const data  = buildCompositionData();
    const blob  = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    const a     = document.createElement('a');
    a.href      = URL.createObjectURL(blob);
    a.download  = (data.title || 'song') + '.json';
    a.click();
}

// Shared by Import Song, Undo/Redo and the shared-link viewer (share.js) --
// replaces the whole editor (title/scale/bpm/pre-roll/lines) with the given
// composition-data object (same shape buildCompositionData() returns).
function applyCompositionData(d) {
    if (d.title)    document.getElementById('song-title').value = d.title;
    if (d.scale) {
        const sv = parseFloat(d.scale) < 256 ? parseFloat(d.scale) * 2 : parseFloat(d.scale);
        let closestOpt = '';
        let minDiff = Infinity;
        document.querySelectorAll('#scale-select option').forEach(opt => {
            const diff = Math.abs(parseFloat(opt.value) - sv);
            if (diff < minDiff) {
                minDiff = diff;
                closestOpt = opt.value;
            }
        });
        document.getElementById('scale-select').value = closestOpt || sv;
    }
    if (d.bpm)      document.getElementById('bpm').value = d.bpm;
    if (d.pre_roll !== undefined)
                    document.getElementById('pre-roll-select').value = d.pre_roll;
    document.getElementById('song-sheet').innerHTML = '';
    (d.lines || []).forEach(line => addLine(line));
}

function loadJsonFile(e) {
    const reader = new FileReader();
    reader.onload = ev => {
        const d = JSON.parse(ev.target.result);
        // Snapshot the pre-import state so Undo can bring it back, then the
        // post-import state, so one Ctrl+Z after an Import reverts it cleanly.
        if (typeof historySnapshotNow === 'function') historySnapshotNow();
        if (typeof window !== 'undefined') window._histSuspend = true;
        applyCompositionData(d);
        if (typeof window !== 'undefined') window._histSuspend = false;
        if (typeof historySnapshotNow === 'function') historySnapshotNow();
    };
    reader.readAsText(e.target.files[0]);
    e.target.value = '';
}

// Event delegation for the entire song-sheet:
//   • note-select change  → apply/remove sustain styling on its note-unit
//   • line-adlib change   → toggle adlib-line class + force-mute tabla
document.getElementById('song-sheet').addEventListener('change', function(e) {

    if (e.target.classList.contains('note-select')) {
        applySustainStyle(e.target.closest('.note-unit'));
        applyRagaHighlights();
    }

    if (e.target.classList.contains('line-adlib')) {
        const lineEl  = e.target.closest('.line-container');
        const isAdlib = e.target.checked;
        lineEl.classList.toggle('adlib-line', isAdlib);
        // Adlib forces percussion to Mute (playback already ignores select when adlib,
        // but switching the UI makes the state visible and consistent).
        const percSel = lineEl.querySelector('.line-perc-sel');
        if (isAdlib && percSel) percSel.value = 'mute';
        // Show or hide the + Beat / − Beat strip
        const beatCtrl = lineEl.querySelector('.adlib-beat-controls');
        if (beatCtrl) beatCtrl.style.display = isAdlib ? 'flex' : 'none';
    }

    // When the user switches the per-line Percussion dropdown, live-update the
    // bol labels in the Beat row so they match the chosen instrument.
    if (e.target.classList.contains('line-perc-sel')) {
        const lineEl  = e.target.closest('.line-container');
        const taalKey = lineEl.getAttribute('data-taal');
        const newBols = e.target.value === 'dholak'
            ? (DholakBols[taalKey] || [])
            : (TaalBols[taalKey]   || []);
        lineEl.querySelectorAll('.bol-text').forEach((el, i) => {
            el.textContent = newBols[i] || '';
        });
    }
});

// ── Adlib + Beat / − Beat buttons ────────────────────────────────────────────
// These appear only on adlib lines; they add/remove a complete beat column
// (lyrics cell + notes cell) so the user can have any number of beats.
document.getElementById('song-sheet').addEventListener('click', function(e) {
    const isAdd = e.target.classList.contains('btn-add-beat');
    const isRem = e.target.classList.contains('btn-rem-beat');
    if (!isAdd && !isRem) return;

    const lineEl = e.target.closest('.line-container');
    const rowL   = lineEl.querySelector('[data-row="lyrics"]');
    const rowN   = lineEl.querySelector('[data-row="notes"]');
    let   beats  = parseInt(lineEl.getAttribute('data-beats')) || 1;

    if (isAdd) {
        // ── Lyrics cell ──
        const lCell = document.createElement('div');
        lCell.className = 'cell';
        lCell.innerHTML = `<input type="text" class="lyr-input lyr" value="">`;
        rowL.appendChild(lCell);

        // ── Notes cell ──
        let _allOpts = '';
        for (const sw in SwaraMap) _allOpts += `<option>${sw}</option>`;

        const nCell = document.createElement('div');
        nCell.className = 'cell';
        nCell.dataset.beatIdx = beats;          // new index

        const sub = document.createElement('div');
        sub.className = 'sub-cell-container';
        sub.appendChild(_makeNoteUnit('-', 0, false, _allOpts, false));

        const addSubBtn = document.createElement('button');
        addSubBtn.className   = 'add-sub-btn';
        addSubBtn.title       = 'Add sub-note (gamak / taan)';
        addSubBtn.textContent = '+';
        addSubBtn.addEventListener('click', () => {
            sub.appendChild(_makeNoteUnit('-', 0, false, _allOpts, true));
            sub.querySelectorAll('.rem-sub-btn').forEach(b => b.style.visibility = '');
        });

        nCell.appendChild(sub);
        nCell.appendChild(addSubBtn);
        rowN.appendChild(nCell);

        beats += 1;

    } else {
        // ── Remove last beat column (min 1) ──
        if (beats <= 1) return;
        const lCells = rowL.querySelectorAll('.cell');
        const nCells = rowN.querySelectorAll('.cell');
        if (lCells.length) lCells[lCells.length - 1].remove();
        if (nCells.length) nCells[nCells.length - 1].remove();
        beats -= 1;
    }

    lineEl.setAttribute('data-beats', beats);
    const counter = lineEl.querySelector('.adlib-beat-count');
    if (counter) counter.textContent = beats + ' beats';
});

// Keep scroll-margin-top on all lines in sync whenever the sticky header resizes
// (toolbar can wrap to 2 rows; raga bar can appear/disappear; ruler changes taal)
new ResizeObserver(updateScrollMargin).observe(document.getElementById('sticky-header'));

// Restore taal / BPM / scale from the last session (no auto-add line on reload)
restoreSettings();

// Persist toolbar settings, refresh ruler, and auto-add first line on taal change
document.getElementById('taal-select').addEventListener('change', function() {
    saveSettings();
    renderTaalRuler(this.value, 'tabla');
    if (document.getElementById('song-sheet').children.length === 0) {
        addLine();
    }
});
document.getElementById('bpm').addEventListener('change', saveSettings);
document.getElementById('scale-select').addEventListener('change', saveSettings);

// Raga Mode controls
document.getElementById('raga-mode-toggle').addEventListener('change', function() {
    const sel = document.getElementById('raga-select');
    sel.style.visibility  = this.checked ? 'visible' : 'hidden';
    sel.style.pointerEvents = this.checked ? '' : 'none';
    applyRagaHighlights();
});
document.getElementById('raga-select').addEventListener('change', applyRagaHighlights);

// NOTE: populateRagaDropdown() is NOT called here — it is called by
// loadRagasFromFirestore() after Firebase init, so that isPremium/isAdmin
// are already declared (avoids Temporal Dead Zone ReferenceError).

