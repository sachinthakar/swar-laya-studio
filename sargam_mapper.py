import numpy as np
import pyaudio
import json
import time
import math
import msvcrt
import sys
import librosa
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
NOTES = {
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81, "F3": 174.61,
    "F#3": 185.00, "G3": 196.00, "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63, "F4": 349.23,
    "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25, "E5": 659.25, "F5": 698.46
}
SCALE_MAP.update(NOTES)

SWARA_NAMES = ["Sa","re","Re","ga","Ga","Ma","ma","Pa","dha","Dha","ni","Ni"]

def yin_pitch(data, sr=44100):
    clean_data = lowpass_filter(data, cutoff=1200, fs=sr)
    rms = np.sqrt(np.mean(clean_data**2))
    if rms < 0.006: return 0
    
    tau_min = int(sr / 1500)
    tau_max = int(sr / 70)
    W = len(data) - tau_max
    if W <= 0: return 0
    
    diff = np.zeros(tau_max)
    for tau in range(1, tau_max):
        diff[tau] = np.mean((clean_data[:-tau_max] - clean_data[tau:tau+W])**2)
        
    cmndf = np.zeros(tau_max)
    cmndf[0] = 1.0
    running_sum = 0.0
    for tau in range(1, tau_max):
        running_sum += diff[tau]
        cmndf[tau] = diff[tau] * tau / (running_sum + 1e-8)
        
    threshold = 0.15
    for tau in range(tau_min, tau_max):
        if cmndf[tau] < threshold:
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            if 0 < tau < tau_max - 1:
                alpha, beta, gamma = cmndf[tau-1], cmndf[tau], cmndf[tau+1]
                denom = 2 * (alpha - 2*beta + gamma + 1e-8)
                if denom != 0:
                    peak = tau + (alpha - gamma) / denom
                    return sr / peak
            return sr / tau
            
    tau = np.argmin(cmndf[tau_min:tau_max]) + tau_min
    if cmndf[tau] < 0.4:
        return sr / tau
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
    Returns the single most prominent note.
    """
    if not buffer: return ["-"], [0]
    
    # Filter out silence
    filtered = [n for n in buffer if n[0] != "-"]
    if not filtered: return ["-"], [0]
    
    # Find the single most common note in the beat
    from collections import Counter
    most_common = Counter(filtered).most_common(1)[0][0]
    
    return [most_common[0]], [most_common[1]]

# --- MAIN RUNTIME ---
try:
    audio_file = input("Enter Audio Filename (e.g. song.wav): ").strip()
    print("Loading audio file... Please wait.")
    y, sr = librosa.load(audio_file, sr=44100, mono=True)
    print("Audio loaded successfully.")
    
    scale_in = input("Enter Scale (e.g. Pandhri 1, C4, or 261.6): ").strip()
    try:
        SA_HZ = float(scale_in)
    except ValueError:
        SA_HZ = SCALE_MAP.get(scale_in, 155.56)
    song_bpm = int(input("Enter BPM of the Song (e.g. 160): "))
    target_bpm = song_bpm
    record_bpm = song_bpm
    adlib_bpm_in = input("Enter Adlib Resolution BPM (e.g. 240): ").strip()
    adlib_bpm = int(adlib_bpm_in) if adlib_bpm_in else 240
    beats_in = input("Enter Taal Beats per Line (e.g. 8 for Keherwa, 7 for Rupak): ").strip()
    beats_per_line = int(beats_in) if beats_in else 8
except Exception as e:
    print(f"Error initializing: {e}")
    sys.exit(1)

CHUNK = 2048 # Increased for YIN algorithm stability
RATE = 44100
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

all_lines = []
curr_line_notes, curr_line_octs = [], []
note_buffer = []
beat_dur = 60.0 / adlib_bpm
mode = "ADLIB"
start_time = time.time()
last_beat_idx = -1
recording_started = False
adlib_silence_threshold = max(2, int(adlib_bpm / 60.0))
adlib_silence_beats = 0

print(f"\n[Recording] Song BPM: {song_bpm} | Adlib Res: {adlib_bpm} | Scale: {scale_in}")
print("Press SPACE for SAM/TAAL mode | ESC to Save & Exit")

try:
    for i in range(0, len(y), CHUNK):
        chunk_data = y[i:i+CHUNK]
        
        # Send audio to speakers
        if len(chunk_data) == CHUNK:
            stream.write(chunk_data.tobytes())
        else:
            # Pad the last chunk with zeros if necessary
            padded = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), 'constant')
            stream.write(padded.tobytes())
            
        freq = yin_pitch(chunk_data)
        swara, octv = get_swara_info(freq, SA_HZ)
        note_buffer.append((swara, octv))
        
        elapsed = time.time() - start_time
        total_beats = int(elapsed / beat_dur)

        # Sync/Mode Switching
        if msvcrt.kbhit():
            key = ord(msvcrt.getch())
            if key == 32: # Space
                # Trim any trailing silence from the Adlib line before switching
                while curr_line_notes and curr_line_notes[-1] == ["-"]:
                    curr_line_notes.pop()
                    curr_line_octs.pop()
                    
                if curr_line_notes and any(n != ["-"] for n in curr_line_notes):
                    all_lines.append({
                        "section": "",
                        "line_instrument": "harmonium",
                        "percussion": "mute",
                        "line_volume": 1.0,
                        "adlib": True,
                        "legato": False,
                        "line_bpm": adlib_bpm,
                        "taal_key": "",
                        "beats": len(curr_line_notes),
                        "lyrics": [[""] for _ in curr_line_notes],
                        "notes": curr_line_notes,
                        "octaves": curr_line_octs,
                        "meend": [[False] * len(b) for b in curr_line_notes]
                    })
                    curr_line_notes, curr_line_octs = [], []
                
                mode = "TAAL"
                beat_dur = 60.0 / song_bpm
                start_time, total_beats, last_beat_idx = time.time(), 0, -1
                print("\n*** SYNCED TO SAM (TAAL MODE) - ADLIB OVER ***")
            elif key == 27: # Esc
                break

        # Process a completed beat
        if total_beats != last_beat_idx:
            beat_notes, beat_octs = process_beat_buffer(note_buffer)
            
            if not recording_started:
                if beat_notes == ["-"]:
                    last_beat_idx = total_beats
                    note_buffer = []
                    continue
                recording_started = True

            curr_line_notes.append(beat_notes)
            curr_line_octs.append(beat_octs)
            
            print(f"Beat {total_beats + 1}: {'-'.join(beat_notes)}      ", end='\r')
            note_buffer = [] # Reset for next beat
            
            # Line breaking logic
            line_complete = False
            if mode == "ADLIB":
                if beat_notes == ["-"]:
                    adlib_silence_beats += 1
                    if adlib_silence_beats >= adlib_silence_threshold and len(curr_line_notes) > adlib_silence_threshold:
                        curr_line_notes = curr_line_notes[:-adlib_silence_threshold]
                        curr_line_octs = curr_line_octs[:-adlib_silence_threshold]
                        line_complete = True
                        adlib_silence_beats = 0
                else:
                    adlib_silence_beats = 0
            elif mode == "TAAL":
                if len(curr_line_notes) >= beats_per_line:
                    line_complete = True

            if line_complete:
                all_lines.append({
                    "section": "",
                    "line_instrument": "harmonium",
                    "percussion": "tabla" if mode == "TAAL" else "mute",
                    "line_volume": 1.0,
                    "adlib": mode == "ADLIB",
                    "legato": False,
                    "line_bpm": adlib_bpm if mode == "ADLIB" else None,
                    "taal_key": "",
                    "beats": len(curr_line_notes),
                    "lyrics": [[""] for _ in curr_line_notes],
                    "notes": curr_line_notes,
                    "octaves": curr_line_octs,
                    "meend": [[False] * len(b) for b in curr_line_notes]
                })
                curr_line_notes, curr_line_octs = [], []
            
            last_beat_idx = total_beats

    # Flush remaining notes at EOF
    if curr_line_notes and any(n != ["-"] for n in curr_line_notes):
        all_lines.append({
            "section": "",
            "line_instrument": "harmonium",
            "percussion": "tabla" if mode == "TAAL" else "mute",
            "line_volume": 1.0,
            "adlib": mode == "ADLIB",
            "legato": mode == "ADLIB",
            "line_bpm": adlib_bpm if mode == "ADLIB" else None,
            "taal_key": "",
            "beats": len(curr_line_notes),
            "lyrics": [[""] for _ in curr_line_notes],
            "notes": curr_line_notes,
            "octaves": curr_line_octs,
            "meend": [[False] * len(b) for b in curr_line_notes]
        })

except KeyboardInterrupt:
    pass

# FINAL OUTPUT CONSTRUCTION
final_output = {
    "title": "Precision Capture",
    "scale": str(SA_HZ),
    "bpm": target_bpm,
    "lines": all_lines
}

with open("precision_output.json", "w") as f:
    json.dump(final_output, f, indent=2)

stream.stop_stream(); stream.close(); p.terminate()
print(f"\nDone! Captured {len(all_lines)} lines.")