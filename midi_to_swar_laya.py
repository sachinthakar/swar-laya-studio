#!/usr/bin/env python3
"""
midi_to_swar_laya.py — MIDI → Swar-Laya Studio JSON Converter
==============================================================
Converts a MIDI file (e.g., exported from basicpitch.spotify.com) into
the Swar-Laya Studio v3 JSON format for import into the sequencer.

INSTALL:
    pip install pretty_midi

BASIC USAGE:
    python midi_to_swar_laya.py song.mid --sa G#4 --title "Sakhi Shejarani" --scale 207.65

SAKHI SHEJARANI (Kali 4):
    python midi_to_swar_laya.py song.mid --sa G#4 --title "Sakhi Shejarani" --scale 207.65 --beats 8 --taal 8

SA NOTE NUMBERS FOR COMMON SCALES:
    Kali 1 = C#4  (MIDI 61)     Pandhri 1 = C4   (MIDI 60)
    Kali 2 = D#4  (MIDI 63)     Pandhri 2 = D4   (MIDI 62)
    Kali 3 = F#4  (MIDI 66)     Pandhri 3 = E4   (MIDI 64)
    Kali 4 = G#4  (MIDI 68)     Pandhri 4 = F4   (MIDI 65)
    Kali 5 = A#4  (MIDI 70)     Pandhri 5 = G4   (MIDI 67)
                                 Pandhri 6 = A4   (MIDI 69)

WORKFLOW (recommended):
    1. Upload MP3 to basicpitch.spotify.com
    2. Download the .mid file
    3. Run this script
    4. Load the JSON into Swar-Laya Studio
    5. Manually correct notes that are wrong (there WILL be errors)
    6. Add lyrics, meend, section names

NOTE: Basic Pitch works best on clean, monophonic recordings.
      On mixed audio (singer + tabla + harmonium), expect errors.
      This script produces a STARTING POINT, not a perfect transcription.
"""

import json
import math
import argparse
import sys
from collections import defaultdict

try:
    import pretty_midi
except ImportError:
    print("ERROR: pretty_midi not installed.")
    print("Run:   pip install pretty_midi")
    sys.exit(1)


# ─── Swar name table ─────────────────────────────────────────────────────────
# Exactly matches the SwaraMap in Swar-Laya Studio v3.
# Key = semitone offset from Sa (0–11)
# Convention: Uppercase = shuddha, lowercase = komal (or tivra for Ma)

SWAR_TABLE = {
    0:  "Sa",
    1:  "re",   # komal Re  (minor 2nd)
    2:  "Re",   # shuddha Re (major 2nd)
    3:  "ga",   # komal Ga  (minor 3rd)
    4:  "Ga",   # shuddha Ga (major 3rd)
    5:  "Ma",   # shuddha Ma (perfect 4th)
    6:  "ma",   # tivra Ma  (tritone)
    7:  "Pa",   # Pancham   (perfect 5th)
    8:  "dha",  # komal Dha (minor 6th)
    9:  "Dha",  # shuddha Dha (major 6th)
    10: "ni",   # komal Ni  (minor 7th)
    11: "Ni",   # shuddha Ni (major 7th)
}

# ─── Note name → MIDI offset within octave ───────────────────────────────────
NOTE_NAME_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_sa(sa_str: str) -> int:
    """
    Parse Sa specification to a MIDI note number.
    Accepts:
      - Integer string: "68"
      - Note name + octave: "G#4", "Ab4", "C5", "D#3"
    """
    sa_str = sa_str.strip()
    try:
        return int(sa_str)
    except ValueError:
        pass

    # Try note name + octave (longest match first to handle "C#", "Db" etc.)
    for name in sorted(NOTE_NAME_MAP.keys(), key=len, reverse=True):
        if sa_str.upper().startswith(name.upper()):
            octave_str = sa_str[len(name):]
            try:
                octave = int(octave_str)
                return (octave + 1) * 12 + NOTE_NAME_MAP[name]
            except ValueError:
                pass

    raise ValueError(
        f"Cannot parse Sa: '{sa_str}'. "
        "Use a MIDI number (e.g. 68) or note name with octave (e.g. G#4, Ab4, C5)."
    )


def midi_to_swar(midi_note: int, sa_midi: int) -> tuple:
    """
    Convert a MIDI pitch to (swar_name, octave).
    octave: 0 = middle (same octave as Sa), 1 = upper, -1 = lower.
    Handles any pitch range correctly.
    """
    diff = midi_note - sa_midi
    octave = 0

    # Shift into 0–11 range while tracking octave
    if diff >= 0:
        octave = diff // 12
        semitone = diff % 12
    else:
        # For negative diff: -1 → octave=-1, semitone=11
        octave = -((-diff - 1) // 12 + 1)
        semitone = diff % 12  # Python % always returns non-negative

    return SWAR_TABLE[semitone], octave


def get_bpm(pm: "pretty_midi.PrettyMIDI") -> int:
    """Extract the first (or only) tempo from MIDI, rounded to integer."""
    _, tempos = pm.get_tempo_changes()
    if len(tempos) > 0:
        return round(float(tempos[0]))
    return 120


def select_track(pm: "pretty_midi.PrettyMIDI", track_index=None):
    """
    Choose which MIDI instrument/track to use.
    - If track_index is given, use that.
    - Otherwise auto-select the non-drum track with the most notes.
    """
    melodic = [inst for inst in pm.instruments if not inst.is_drum and len(inst.notes) > 0]

    if not melodic:
        return None

    if track_index is not None:
        all_tracks = pm.instruments
        if track_index >= len(all_tracks):
            print(f"WARNING: Track index {track_index} out of range ({len(all_tracks)} tracks). "
                  f"Using auto-selection.")
        else:
            return all_tracks[track_index]

    # Auto: most notes
    return max(melodic, key=lambda x: len(x.notes))


# ─── Core conversion ──────────────────────────────────────────────────────────

def quantize_notes(notes_raw, bpm: int, max_sub_beats: int = 4,
                   start_time: float = 0.0) -> dict:
    """
    Assign each MIDI note to a beat index and sub-beat slot.

    notes_raw: list of (start_sec, end_sec, midi_pitch, velocity)
    Returns:   dict { beat_index: [(swar, octave), ...] }
               Each beat list is sorted by time, capped at max_sub_beats.

    Strategy when a beat is crowded (many overlapping notes from polyphonic
    detection): keep only the max_sub_beats highest-velocity notes, then
    re-sort by time so the JSON reads left-to-right in time order.
    """
    beat_dur = 60.0 / bpm
    raw_buckets = defaultdict(list)  # beat_idx → [(start, swar, octave, vel)]

    for (start, end, pitch, vel, swar, octave) in notes_raw:
        adjusted = start - start_time
        if adjusted < 0:
            continue   # skip notes before the start time
        beat_idx = int(adjusted / beat_dur)
        raw_buckets[beat_idx].append((adjusted, swar, octave, vel))

    result = {}
    for beat_idx, items in raw_buckets.items():
        if max_sub_beats == 1:
            # Single-note mode: pick the loudest note in the beat.
            # Basic Pitch often detects harmonics as extra notes; the loudest
            # one is most likely the true melody note.
            best = max(items, key=lambda x: x[3])
            result[beat_idx] = [(best[1], best[2])]
        else:
            # Multi-note mode: sort by time, cap at max_sub_beats.
            items.sort(key=lambda x: x[0])
            if len(items) > max_sub_beats:
                # Too many notes: keep highest-velocity ones, then re-sort by time.
                items = sorted(items, key=lambda x: -x[3])[:max_sub_beats]
                items.sort(key=lambda x: x[0])
            result[beat_idx] = [(s, o) for (_, s, o, _) in items]

    return result


def build_lines(beat_dict: dict, bpm: int, beats_per_line: int,
                taal_key: str, instrument: str) -> list:
    """
    Pack quantized beat data into line dicts matching the Swar-Laya JSON schema.
    Beats with no notes get a rest marker "-".
    """
    if not beat_dict:
        return []

    max_beat = max(beat_dict.keys())
    num_lines = math.ceil((max_beat + 1) / beats_per_line)
    lines = []

    for line_num in range(num_lines):
        line_start = line_num * beats_per_line
        notes_grid   = []
        octaves_grid = []
        meend_grid   = []
        lyrics_list  = []

        for b in range(beats_per_line):
            beat_idx  = line_start + b
            beat_data = beat_dict.get(beat_idx)

            if not beat_data:
                # Rest / empty beat
                notes_grid.append(["-"])
                octaves_grid.append([0])
                meend_grid.append([False])
                lyrics_list.append("-")
            else:
                swars  = [d[0] for d in beat_data]
                octs   = [d[1] for d in beat_data]
                meends = [False] * len(swars)
                notes_grid.append(swars)
                octaves_grid.append(octs)
                meend_grid.append(meends)
                lyrics_list.append("-")         # blank: user fills in actual lyrics

        lines.append({
            "section":         f"Line {line_num + 1}",
            "line_instrument": instrument,
            "tabla_mute":      False,
            "line_volume":     1.0,
            "taal_key":        taal_key,
            "beats":           str(beats_per_line),
            "lyrics":          lyrics_list,
            "notes":           notes_grid,
            "octaves":         octaves_grid,
            "meend":           meend_grid,
        })

    return lines


def convert(midi_path: str, sa_midi: int, beats_per_line: int,
            taal_key: str, title: str, scale: str, instrument: str,
            track_index=None, max_sub_beats: int = 1,
            bpm_override: int = None, start_time: float = 0.0) -> dict:
    """
    Full pipeline: MIDI file → Swar-Laya Studio JSON dict.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    # ── Report what's in the file ──────────────────────────────────────────
    bpm = bpm_override if bpm_override else get_bpm(pm)
    print(f"\n{'-'*50}")
    print(f"  MIDI file : {midi_path}")
    print(f"  Duration  : {pm.get_end_time():.2f} s")
    print(f"  BPM       : {bpm}  {'(from MIDI)' if not bpm_override else '(overridden)'}")
    print(f"  Sa        : MIDI {sa_midi}  = {midi_to_swar(sa_midi, sa_midi)[0]}")
    print(f"\n  Tracks in file:")
    for i, inst in enumerate(pm.instruments):
        tag = " <- auto-selected" if (not inst.is_drum and len(inst.notes) > 0 and
              inst == max((x for x in pm.instruments
                           if not x.is_drum and x.notes), key=lambda x: len(x.notes),
                          default=None)) else ""
        print(f"    [{i}] '{inst.name or 'unnamed'}' - {len(inst.notes)} notes"
              f"{'  (drum)' if inst.is_drum else ''}{tag}")
    print(f"{'-'*50}\n")

    # ── Select track ───────────────────────────────────────────────────────
    selected = select_track(pm, track_index)
    if selected is None:
        print("ERROR: No melodic notes found in MIDI file.")
        return None
    print(f"Using track: '{selected.name or 'unnamed'}' ({len(selected.notes)} notes)")

    # ── Collect and annotate notes ─────────────────────────────────────────
    notes_raw = []
    for note in selected.notes:
        swar, octave = midi_to_swar(note.pitch, sa_midi)
        notes_raw.append((note.start, note.end, note.pitch,
                           note.velocity, swar, octave))

    # ── Quantize to beat grid ──────────────────────────────────────────────
    if start_time > 0:
        print(f"Start time offset: {start_time:.2f} s (notes before this are skipped)")
    beat_dict = quantize_notes(notes_raw, bpm, max_sub_beats, start_time=start_time)

    # ── Build JSON lines ───────────────────────────────────────────────────
    lines = build_lines(beat_dict, bpm, beats_per_line, taal_key, instrument)

    return {
        "title": title,
        "scale": scale,
        "bpm":   bpm,
        "lines": lines,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def print_first_line_preview(result: dict):
    """Print a readable preview of the first line to stdout."""
    if not result or not result["lines"]:
        return
    first = result["lines"][0]
    print("\nFirst line preview:")
    print(f"  {'Beat':<6}  {'Notes':<20}  {'Octaves'}")
    print(f"  {'----':<6}  {'-----':<20}  {'-------'}")
    for i, (notes, octs) in enumerate(zip(first["notes"], first["octaves"])):
        oct_labels = ["^" if o == 1 else "v" if o == -1 else " " for o in octs]
        note_str   = "  ".join(f"{n}{l}" for n, l in zip(notes, oct_labels))
        print(f"  {i+1:<6}  {note_str}")


def main():
    parser = argparse.ArgumentParser(
        prog="midi_to_swar_laya",
        description="Convert Basic Pitch MIDI to Swar-Laya Studio v3 JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("midi_file",
        help="Input MIDI file path (.mid or .midi)")

    parser.add_argument("-o", "--output",
        default=None,
        help="Output JSON file path (default: same directory, .json extension)")

    parser.add_argument("--sa",
        default="68",
        metavar="NOTE",
        help="Sa note — MIDI number or name+octave. "
             "Default: 68 (G#4, Kali 4). "
             "Examples: 68  G#4  Ab4  C5  D#4")

    parser.add_argument("--beats",
        type=int, default=8,
        metavar="N",
        help="Beats per line / taal avart. Default: 8")

    parser.add_argument("--taal",
        default="8",
        metavar="KEY",
        help="Taal key: 4 | 6 | 7 | 8 | 8_bhajani | 12 | 16. Default: '8'")

    parser.add_argument("--title",
        default="Untitled",
        help="Song title for the JSON header. Default: 'Untitled'")

    parser.add_argument("--scale",
        default="207.65",
        metavar="HZ",
        help="Sa frequency in Hz for JSON (scale field). "
             "Default: 207.65 (Kali 4 = G#3, plays at G#4). "
             "See scale selector in Swar-Laya Studio.")

    parser.add_argument("--instrument",
        default="harmonium",
        choices=["harmonium", "flute", "sitar", "violin", "piano"],
        help="Line instrument. Default: harmonium")

    parser.add_argument("--track",
        type=int, default=None,
        metavar="INDEX",
        help="MIDI track index to use (0-based). "
             "Default: auto-select track with most notes. "
             "Run once first to see track listing.")

    parser.add_argument("--max-sub-beats",
        type=int, default=1,
        metavar="N",
        help="Max sub-beats per beat. Default: 1 (one dominant note per beat). "
             "Use 2–4 only for fast passages (taan) where you trust the MIDI quality.")

    parser.add_argument("--bpm",
        type=int, default=None,
        metavar="N",
        help="Override BPM from MIDI (useful if tempo detection is wrong)")

    parser.add_argument("--start-time",
        type=float, default=0.0,
        metavar="SEC",
        help="Skip notes before this time (seconds). Beat grid starts here. "
             "Use this to align the taal grid with the first beat of the melody. "
             "Example: --start-time 15.15 for a song where singing starts at 15s.")

    args = parser.parse_args()

    # ── Parse Sa ──────────────────────────────────────────────────────────
    try:
        sa_midi = parse_sa(args.sa)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # ── Run conversion ────────────────────────────────────────────────────
    result = convert(
        midi_path      = args.midi_file,
        sa_midi        = sa_midi,
        beats_per_line = args.beats,
        taal_key       = args.taal,
        title          = args.title,
        scale          = args.scale,
        instrument     = args.instrument,
        track_index    = args.track,
        max_sub_beats  = args.max_sub_beats,
        bpm_override   = args.bpm,
        start_time     = args.start_time,
    )

    if not result:
        sys.exit(1)

    # ── Determine output path ─────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        base = args.midi_file
        for ext in (".mid", ".midi", ".MID", ".MIDI"):
            if base.endswith(ext):
                base = base[:-len(ext)]
                break
        out_path = base + ".json"

    # ── Write JSON ────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved  : {out_path}")
    print(f"Title  : {result['title']}")
    print(f"BPM    : {result['bpm']}")
    print(f"Lines  : {len(result['lines'])}")

    print_first_line_preview(result)

    print("\nNEXT STEPS (manual corrections needed):")
    print("   1. Load JSON in Swar-Laya Studio")
    print("   2. Play alongside the original MP3")
    print("   3. Fix wrong notes beat by beat")
    print("   4. Replace lyric placeholders with actual lyrics")
    print("   5. Add meend (glide) where needed")
    print("   6. Set correct section names (Mukhda / Antara)")
    print("   7. Adjust octave markers (up/down) where notes are in wrong octave")


if __name__ == "__main__":
    main()
