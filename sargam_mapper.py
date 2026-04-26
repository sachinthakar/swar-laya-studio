import numpy as np
import pyaudio
import json
import time
import math
import msvcrt
from scipy.signal import butter, lfilter

# --- DSP FILTERS ---
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff=1100, fs=44100, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, data)

# --- CONFIG ---
SCALE_MAP = {
    "Pandhri 1": 130.815, "Kali 1": 138.59, "Pandhri 2": 146.835, "Kali 2": 155.565,
    "Pandhri 3": 164.815, "Pandhri 4": 174.615, "Kali 3": 185.00, "Pandhri 5": 196.00,
    "Kali 4": 207.65, "Pandhri 6": 220.00, "Kali 5": 233.08, "Pandhri 7": 246.94
}
SWARA_NAMES = ["Sa","re","Re","ga","Ga","Ma","ma","Pa","dha","Dha","ni","Ni"]

def amdf_pitch(data, sr=44100):
    clean_data = lowpass_filter(data, cutoff=1200, fs=sr)
    rms = np.sqrt(np.mean(clean_data**2))
    if rms < 0.006: return 0
    
    min_p, max_p = int(sr/1000), int(sr/70)
    amdf = np.zeros(max_p)
    for tau in range(min_p, max_p):
        amdf[tau] = np.mean(np.abs(clean_data[tau:] - clean_data[:-tau]))
    
    valley = np.argmin(amdf[min_p:]) + min_p
    if amdf[valley] < np.mean(amdf) * 0.45:
        return sr / valley
    return 0

def get_swara_info(freq, sa_hz):
    if freq < (sa_hz * 0.5): return "-", 0
    try:
        semitones = 12 * math.log2(freq / sa_hz)
        index = int(round(semitones))
        return SWARA_NAMES[index % 12], index // 12
    except:
        return "-", 0

def process_beat_buffer(buffer):
    """
    Analyzes a buffer of detected pitches within one beat.
    Identifies multiple notes if the player plays quickly (e.g., Sa-Re-Ga-Ma).
    """
    if not buffer: return ["-"], [0]
    
    # Filter out silence and group consecutive identical notes
    filtered = [n for n in buffer if n[0] != "-"]
    if not filtered: return ["-"], [0]
    
    compact_notes = []
    compact_octs = []
    
    if filtered:
        curr_n, curr_o = filtered[0]
        count = 1
        # Minimum samples to consider it a "real" note (denoising)
        min_samples = 2 
        
        for i in range(1, len(filtered)):
            if filtered[i] == (curr_n, curr_o):
                count += 1
            else:
                if count >= min_samples:
                    compact_notes.append(curr_n)
                    compact_octs.append(curr_o)
                curr_n, curr_o = filtered[i]
                count = 1
        # Add the last one
        if count >= min_samples:
            compact_notes.append(curr_n)
            compact_octs.append(curr_o)

    # If no stable notes found, return silence
    if not compact_notes: return ["-"], [0]
    return compact_notes, compact_octs

# --- MAIN RUNTIME ---
try:
    scale_in = input("Enter Scale: ").strip()
    SA_HZ = SCALE_MAP.get(scale_in, 155.56)
    BPM = int(input("Enter BPM: "))
except:
    SA_HZ, BPM = 155.56, 120

CHUNK = 1024 # Smaller chunk for higher temporal resolution
RATE = 44100
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

all_lines = []
curr_line_notes, curr_line_octs = [], []
note_buffer = []
beat_dur = 60.0 / BPM
mode = "ADLIB"
start_time = time.time()
last_beat_idx = -1

print(f"\n[Recording] BPM: {BPM} | Scale: {scale_in}")
print("Press SPACE for SAM/TAAL mode | ESC to Save & Exit")

try:
    while True:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
        freq = amdf_pitch(data)
        swara, octv = get_swara_info(freq, SA_HZ)
        note_buffer.append((swara, octv))
        
        elapsed = time.time() - start_time
        total_beats = int(elapsed / beat_dur)

        # Sync/Mode Switching
        if msvcrt.kbhit():
            key = ord(msvcrt.getch())
            if key == 32: # Space
                mode = "TAAL"
                start_time, total_beats, last_beat_idx = time.time(), 0, -1
                print("\n*** SYNCED TO SAM ***")
            elif key == 27: # Esc
                break

        # Process a completed beat
        if total_beats != last_beat_idx:
            beat_notes, beat_octs = process_beat_buffer(note_buffer)
            curr_line_notes.append(beat_notes)
            curr_line_octs.append(beat_octs)
            
            print(f"Beat {total_beats + 1}: {'-'.join(beat_notes)}      ", end='\r')
            note_buffer = [] # Reset for next beat
            
            # Line breaking logic (every 8 beats)
            if len(curr_line_notes) >= 8:
                all_lines.append({
                    "section": "",
                    "line_instrument": "harmonium",
                    "percussion": "tabla" if mode == "TAAL" else "mute",
                    "line_volume": 1.0,
                    "adlib": mode == "ADLIB",
                    "legato": False,
                    "line_bpm": None,
                    "taal_key": "",
                    "beats": len(curr_line_notes),
                    "lyrics": [[""] for _ in curr_line_notes],
                    "notes": curr_line_notes,
                    "octaves": curr_line_octs,
                    "meend": [[False] * len(b) for b in curr_line_notes]
                })
                curr_line_notes, curr_line_octs = [], []
            
            last_beat_idx = total_beats

except KeyboardInterrupt:
    pass

# FINAL OUTPUT CONSTRUCTION
final_output = {
    "title": "Precision Capture",
    "scale": str(SA_HZ),
    "bpm": BPM,
    "lines": all_lines
}

with open("precision_output.json", "w") as f:
    json.dump(final_output, f, indent=2)

stream.stop_stream(); stream.close(); p.terminate()
print(f"\nDone! Captured {len(all_lines)} lines.")