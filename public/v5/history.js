// Swar-Laya Studio v5 — history: Undo/Redo for the composition
// (title, scale, bpm, pre-roll, and every line's notes/lyrics/settings)
//
// Design: rather than hook every individual edit handler (dozens of inputs/
// selects/buttons across the grid), this takes a periodic full snapshot of
// buildCompositionData() (sheet.js) -- cheap, since a composition is a small
// JSON object -- whenever a broad "something in the toolbar or sheet changed"
// signal fires (input/change/click, delegated from the two containers that
// hold every relevant control). Duplicate snapshots are skipped by comparing
// serialized JSON, so clicks/edits that don't actually change the data (e.g.
// clicking ▶ Line, or a slider that snaps back to the same value) are no-ops
// -- this makes broad, cheap delegation safe instead of needing one precise
// hook per mutation site.
//
// window._histSuspend is also set by sheet.js/cloud.js around their own
// destructive loads (Import Song, Load from Cloud, New Composition) so that
// rebuilding the whole sheet from JSON doesn't get recorded as dozens of
// spurious history entries -- those call historySnapshotNow() themselves,
// once before and once after, to bracket the load as a single undo step.

let _histStack   = [JSON.stringify(buildCompositionData())];
let _histIndex   = 0;
let _histTimer   = null;
window._histSuspend = false;   // read/set from sheet.js and cloud.js too
const HIST_LIMIT = 60;

function _historySnapshot() {
    if (window._histSuspend) return;
    const snap = JSON.stringify(buildCompositionData());
    if (_histStack[_histIndex] === snap) return;   // nothing actually changed
    _histStack = _histStack.slice(0, _histIndex + 1);
    _histStack.push(snap);
    _histIndex++;
    if (_histStack.length > HIST_LIMIT) {
        _histStack.shift();
        _histIndex--;
    }
}

/** Debounced snapshot -- used for ordinary typing/dragging so a fast run of
 *  keystrokes becomes one history entry, not one per keystroke. */
function scheduleHistorySnapshot() {
    if (window._histSuspend) return;
    clearTimeout(_histTimer);
    _histTimer = setTimeout(_historySnapshot, 500);
}

/** Immediate (non-debounced) snapshot -- used to bracket a destructive load
 *  (Import Song, Load from Cloud, New Composition) as its own undo step. */
function historySnapshotNow() {
    clearTimeout(_histTimer);
    _historySnapshot();
}

function _restoreHistoryEntry() {
    window._histSuspend = true;
    applyCompositionData(JSON.parse(_histStack[_histIndex]));
    window._histSuspend = false;
}

function historyUndo() {
    clearTimeout(_histTimer);
    _historySnapshot();   // capture any pending debounced edit first, so it isn't lost
    if (_histIndex <= 0) { showToast('Nothing to undo'); return; }
    _histIndex--;
    _restoreHistoryEntry();
    showToast('Undo ✓');
}

function historyRedo() {
    if (_histIndex >= _histStack.length - 1) { showToast('Nothing to redo'); return; }
    _histIndex++;
    _restoreHistoryEntry();
    showToast('Redo ✓');
}

// ── Wiring ────────────────────────────────────────────────────────────────
// Ctrl/Cmd+Z = Undo, Ctrl/Cmd+Shift+Z or Ctrl/Cmd+Y = Redo (both common
// conventions, so either works no matter which the artist is used to).
document.addEventListener('keydown', e => {
    const mod = e.ctrlKey || e.metaKey;
    if (!mod) return;
    const key = e.key.toLowerCase();
    if (key === 'z' && !e.shiftKey) {
        e.preventDefault();
        historyUndo();
    } else if ((key === 'z' && e.shiftKey) || key === 'y') {
        e.preventDefault();
        historyRedo();
    }
});

// Delegated listeners: the sticky header (song title, scale, taal, bpm,
// pre-roll, volumes...) and the song sheet (every line's notes, lyrics and
// per-line settings) cover every control that can change the composition.
['sticky-header', 'song-sheet'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input',  scheduleHistorySnapshot);
    el.addEventListener('change', scheduleHistorySnapshot);
    el.addEventListener('click',  scheduleHistorySnapshot);
});
