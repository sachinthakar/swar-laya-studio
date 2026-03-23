#!/usr/bin/env python3
"""
split_samples.py — Split instrument master MP3 files into individual note files.

Each note gets its own file: C2.mp3, Cs2.mp3, D2.mp3, Ds2.mp3 … (Cs = C#, Ds = D#, etc.)
Output goes directly into: public/samples/{instrument}/

Requirements:
    pip install pydub
    ffmpeg must be installed — https://ffmpeg.org/download.html
    (on Windows: winget install Gyan.FFmpeg  OR  choco install ffmpeg)

Usage:
    python split_samples.py

After running:
    git add public/samples/
    git commit -m "Add individual note samples"
    git push
"""

import os, re
from pydub import AudioSegment

# ── Configuration — adjust paths / timings as needed ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INSTRUMENTS = {
    'violin': {
        'file':         os.path.join(BASE_DIR, 'Sound Samples', 'Violin-SamplesC2-C5.mp3'),
        'noteDuration': 3.0,   # seconds per note
        'startOffset':  1.5,   # seconds of silence/blank before the FIRST note begins
        'startNote':    'C2',  # note name assigned to the first slice
    },
    'flute': {
        'file':         os.path.join(BASE_DIR, 'Sound Samples', 'Flute-SamplesC2-C5.mp3'),
        'noteDuration': 3.0,
        'startOffset':  0.0,
        'startNote':    'C3',  # flute can't play C2; recording starts at C3
    },
    'harmonium': {
        # Target range: G2→C6 (42 chromatic notes, zero pitch shifting in app)
        # If recording all notes individually, record G2, Gs2, A2, As2, B2,
        # then C3 through C6 chromatically (37 more notes) = 42 files total.
        # If using a master file that starts at C2, set startNote='C2'; notes
        # below C2 (G2–B2) must be recorded individually and placed manually.
        'file':         os.path.join(BASE_DIR, 'Sound Samples', 'Harmonium-SamplesG2-C6.mp3'),
        'noteDuration': 3.0,
        'startOffset':  0.0,
        'startNote':    'G2',
    },
}

# ── Note name helpers ─────────────────────────────────────────────────────────
CHROMATIC = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']

def build_note_range(start_note: str, count: int) -> list[str]:
    m = re.match(r'^([A-Gs]+)(\d+)$', start_note)
    if not m:
        raise ValueError(f'Invalid note name: {start_note!r}  (expected e.g. C2, Cs3, A4)')
    ni = CHROMATIC.index(m.group(1))
    oc = int(m.group(2))
    result = []
    for _ in range(count):
        result.append(CHROMATIC[ni] + str(oc))
        ni += 1
        if ni >= 12:
            ni = 0
            oc += 1
    return result

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    any_done = False

    for inst, cfg in INSTRUMENTS.items():
        path = cfg['file']
        if not os.path.exists(path):
            print(f'\n[{inst}] ⚠  File not found — skipping:\n   {path}')
            continue

        print(f'\n[{inst}] Loading {os.path.basename(path)} …')
        audio  = AudioSegment.from_file(path)
        dur_ms = int(cfg['noteDuration'] * 1000)
        off_ms = int(cfg['startOffset']  * 1000)

        # Auto-detect note count from remaining audio length
        remaining_ms = len(audio) - off_ms
        count = min(remaining_ms // dur_ms, 48)   # cap at 4 octaves (48 semitones)

        notes   = build_note_range(cfg['startNote'], count)
        out_dir = os.path.join(BASE_DIR, 'public', 'samples', inst)
        os.makedirs(out_dir, exist_ok=True)

        print(f'  File length : {len(audio)/1000:.1f}s')
        print(f'  Start offset: {off_ms/1000:.1f}s')
        print(f'  Note duration: {dur_ms/1000:.1f}s  ×  {count} notes')
        print(f'  Note range  : {notes[0]} → {notes[-1]}')
        print(f'  Output dir  : {out_dir}')
        print()

        written = 0
        for i, note in enumerate(notes):
            start_ms = off_ms + i * dur_ms
            end_ms   = start_ms + dur_ms
            if end_ms > len(audio):
                print(f'  ⚠  {note}: end ({end_ms}ms) past file end — stopping at {written} notes.')
                break

            chunk    = audio[start_ms:end_ms]
            out_path = os.path.join(out_dir, f'{note}.mp3')
            # Export mono 128 kbps — plenty for a single sustained note
            chunk.set_channels(1).export(out_path, format='mp3', bitrate='128k')
            print(f'  ✓  {note}.mp3   ({start_ms/1000:.2f}s – {end_ms/1000:.2f}s)')
            written += 1

        print(f'\n  [{inst}] Done — {written} files written.')
        any_done = True

    if any_done:
        print('\n' + '─' * 60)
        print('✅  All done!  Next steps:')
        print()
        print('  1. Listen to a few output files to verify note accuracy.')
        print('     If a note is cut slightly early/late, adjust noteDuration')
        print('     or startOffset in INSTRUMENTS config above and re-run.')
        print()
        print('  2. Commit and push the new files:')
        print('       git add public/samples/')
        print('       git commit -m "Add individual note samples (violin/flute/harmonium)"')
        print('       git push')
        print('─' * 60)
    else:
        print('\n❌  No files were processed. Check the file paths in INSTRUMENTS config.')


if __name__ == '__main__':
    main()
