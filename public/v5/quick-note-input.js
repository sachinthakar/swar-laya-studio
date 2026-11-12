// Swar-Laya Studio — Quick Note Input (fast-entry layer)
// ---------------------------------------------------------------------------
// Loaded AFTER sheet.js. Does not change sheet.js, engine.js or the existing
// <select class="note-select">/.octave-btn/.meend-btn markup in any way —
// it only adds extra, optional ways to drive them from the keyboard/wheel so
// a composer rarely has to open the note dropdown or hunt for a small button.
//
// RULE OF THUMB: every key below acts on whichever beat is currently
// focused. Moving which beat is focused is a separate, explicit step
// (Tab / Enter / Left / Right arrow). This keeps "set a value" and "move"
// from fighting each other once octave/meend come into play.
//
//   SET THE NOTE (focused beat) — letter case matches the notation's own
//   convention: lowercase = komal/tivra, UPPERCASE = shuddha.
//     s/S = Sa   r = re   R = Re   g = ga   G = Ga
//     m = ma(tivra)  M = Ma(shuddha)   p/P = Pa
//     d = dha   D = Dha   n = ni   N = Ni
//     - (hyphen) or Space = rest        ~ = sustain (hold)
//
//   OCTAVE of the focused beat (mandra/madhya/taar/atitar):
//     .  one step up    (madhya → taar → atitar)
//     ,  one step down  (madhya → mandra → mandra-2)
//     >  (Shift+.) jump straight to atitar   (press again to clear)
//     <  (Shift+,) jump straight to mandra-2 (press again to clear)
//
//   MEEND (glide) on the focused beat:
//     '  (apostrophe) toggles meend on/off
//
//   MOVE which beat is focused (does not change its value):
//     Tab / Shift+Tab, Enter, Right arrow / Left arrow
//     (Up/Down arrow are left to the browser's own "step through the
//      dropdown's options" behavior)
//
//   SUB-NOTES (gamak/taan — more than one note inside the same beat):
//     +  or  =   adds a sub-note to the focused beat and jumps into it
//     Delete     removes the focused sub-note (a beat always keeps >=1)
//
//   MOUSE WHEEL over any beat — cycles its note without clicking anything
//   open (works whether or not that beat is focused).
//
// The native dropdown, octave buttons and meend button are left fully in
// place as a visual readout and mouse fallback.
// ---------------------------------------------------------------------------

(function () {

    // ── Legend hide/show (persisted so a user who's learned the shortcuts
    //    doesn't see the box again on reload) ──
    (function setupLegendToggle() {
        const KEY     = 'sl_qni_legend_hidden';
        const legend  = document.getElementById('qni-legend');
        const hideBtn = document.getElementById('qni-legend-hide');
        const showBtn = document.getElementById('qni-legend-show');
        if (!legend || !hideBtn || !showBtn) return;   // markup not present on this page

        function setHidden(hidden) {
            legend.style.display  = hidden ? 'none' : '';
            showBtn.style.display = hidden ? '' : 'none';
            try { localStorage.setItem(KEY, hidden ? '1' : '0'); } catch (err) { /* ignore */ }
        }

        let startHidden = false;
        try { startHidden = localStorage.getItem(KEY) === '1'; } catch (err) { /* ignore */ }
        setHidden(startHidden);

        hideBtn.addEventListener('click', () => setHidden(true));
        showBtn.addEventListener('click', () => setHidden(false));
    })();


    // Letter you type -> exact option value already used by SwaraMap/sheet.js.
    const NOTE_KEYS = {
        's': 'Sa',  'S': 'Sa',
        'r': 're',  'R': 'Re',
        'g': 'ga',  'G': 'Ga',
        'm': 'ma',  'M': 'Ma',
        'p': 'Pa',  'P': 'Pa',
        'd': 'dha', 'D': 'Dha',
        'n': 'ni',  'N': 'Ni',
        '-': '-',   '_': '-',
        '~': '~'
    };

    // Fixed left-to-right order for wheel-cycling. Matches the order
    // options already appear in the dropdown (SwaraMap key order).
    let NOTE_ORDER = null;
    function noteOrder() {
        if (!NOTE_ORDER) {
            const probe = document.querySelector('.note-select');
            NOTE_ORDER = probe
                ? Array.from(probe.options).map(o => o.value)
                : ['-', '~', 'Sa', 're', 'Re', 'ga', 'Ga', 'Ma', 'ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni'];
        }
        return NOTE_ORDER;
    }

    function allSelects() {
        return Array.from(document.querySelectorAll('#song-sheet .note-select'));
    }

    function setNoteValue(sel, value) {
        if (!sel || sel.value === value) return;
        const has = Array.from(sel.options).some(o => o.value === value);
        if (!has) return;
        sel.value = value;
        sel.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function moveFocus(sel, dir) {
        const list = allSelects();
        const idx  = list.indexOf(sel);
        if (idx === -1) return;
        const target = list[idx + dir];
        if (target) target.focus();
    }

    // ── Octave (reuses sheet.js's own toggleOctave so the exclusive-select
    //    and "click again to clear" behavior is identical to the mouse path) ──
    function currentOctave(unit) {
        const active = unit.querySelector('.octave-btn.active');
        return active ? parseInt(active.dataset.oct, 10) : 0;
    }
    function octaveBtn(unit, val) {
        return unit.querySelector(`.octave-btn[data-oct="${val}"]`);
    }
    function stepOctave(unit, dir) {
        if (!unit) return;
        const cur = currentOctave(unit);
        let target = cur + dir;
        if (target > 2)  target = 2;
        if (target < -2) target = -2;
        if (target === cur) return;
        applyOctave(unit, target, cur);
    }
    function jumpOctave(unit, val) {
        if (!unit) return;
        const cur = currentOctave(unit);
        applyOctave(unit, cur === val ? 0 : val, cur);
    }
    function applyOctave(unit, target, cur) {
        if (target === 0) {
            const active = unit.querySelector('.octave-btn.active');
            if (active && !active.disabled) toggleOctave(active, cur); // clears it
            return;
        }
        const btn = octaveBtn(unit, target);
        if (btn && !btn.disabled) toggleOctave(btn, target);
    }
    function toggleMeend(unit) {
        if (!unit) return;
        const btn = unit.querySelector('.meend-btn');
        if (btn && !btn.disabled) btn.classList.toggle('active');
    }

    // ── Sub-notes (gamak/taan) — reuses the existing "+" button and the
    //    existing global removeSubNote() from sheet.js, so behavior/state
    //    stays identical to the mouse path. ──
    function addSubNoteViaKeyboard(sel) {
        const cell = sel.closest('.cell');
        if (!cell) return;
        const addBtn = cell.querySelector('.add-sub-btn');
        if (!addBtn) return;
        addBtn.click();
        const units = cell.querySelectorAll('.note-select');
        const last  = units[units.length - 1];
        if (last) last.focus();
    }
    function removeSubNoteViaKeyboard(sel) {
        const unit = sel.closest('.note-unit');
        const cell = sel.closest('.cell');
        if (!unit || !cell) return;
        const remBtn = unit.querySelector('.rem-sub-btn');
        if (!remBtn) return;
        removeSubNote(remBtn);                 // no-ops if this is the last sub-note
        const firstSel = cell.querySelector('.note-select');
        if (firstSel) firstSel.focus();
    }

    document.getElementById('song-sheet').addEventListener('keydown', function (e) {
        if (!e.target.classList.contains('note-select')) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;   // don't hijack OS/browser shortcuts

        const unit = e.target.closest('.note-unit');

        if (Object.prototype.hasOwnProperty.call(NOTE_KEYS, e.key)) {
            e.preventDefault();
            setNoteValue(e.target, NOTE_KEYS[e.key]);
            return;
        }
        if (e.key === ' ') { e.preventDefault(); setNoteValue(e.target, '-'); return; }
        if (e.key === 'Backspace') { e.preventDefault(); setNoteValue(e.target, '-'); return; }

        if (e.key === '.') { e.preventDefault(); stepOctave(unit, +1); return; }
        if (e.key === ',') { e.preventDefault(); stepOctave(unit, -1); return; }
        if (e.key === '>') { e.preventDefault(); jumpOctave(unit, 2);  return; }
        if (e.key === '<') { e.preventDefault(); jumpOctave(unit, -2); return; }
        if (e.key === "'") { e.preventDefault(); toggleMeend(unit);   return; }
        if (e.key === '+' || e.key === '=') { e.preventDefault(); addSubNoteViaKeyboard(e.target);    return; }
        if (e.key === 'Delete')              { e.preventDefault(); removeSubNoteViaKeyboard(e.target); return; }

        if (e.key === 'Enter') { e.preventDefault(); moveFocus(e.target, +1); return; }
        if (e.key === 'Tab')   { e.preventDefault(); moveFocus(e.target, e.shiftKey ? -1 : +1); return; }
        if (e.key === 'ArrowRight') { e.preventDefault(); moveFocus(e.target, +1); return; }
        if (e.key === 'ArrowLeft')  { e.preventDefault(); moveFocus(e.target, -1); return; }
        // ArrowUp / ArrowDown left to the browser's native dropdown stepping.
    });

    // Mouse wheel over any note cell cycles its value — no click/open needed.
    document.getElementById('song-sheet').addEventListener('wheel', function (e) {
        const sel = e.target.closest ? e.target.closest('.note-select') : null;
        if (!sel) return;
        e.preventDefault();
        const order = noteOrder();
        const cur   = order.indexOf(sel.value);
        const step  = e.deltaY > 0 ? 1 : -1;
        let next    = (cur === -1 ? 0 : cur) + step;
        if (next < 0) next = order.length - 1;
        if (next >= order.length) next = 0;
        setNoteValue(sel, order[next]);
    }, { passive: false });

})();
