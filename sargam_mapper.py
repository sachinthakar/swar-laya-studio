import numpy as np
import pyaudio
import json
import time
import math
import msvcrt
import sys
import librosa
from scipy.signal import butter, lfilter
import requests
import copy

GEMINI_API_KEY = "AIzaSyA400Elj7amr8zq4uBC4Xw3CJGNl2irnm0"

def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"
        }
    }
    
    print("Sending data to Gemini API... (This may take a minute)")
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
        return None
        
    res_json = response.json()
    try:
        content = res_json['candidates'][0]['content']['parts'][0]['text']
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        return json.loads(content)
    except Exception as e:
        print("Failed to parse Gemini response:", e)
        return None

MISTRAL_API_KEY = "WaeawLQ7ysDSf0nCae1pBhE11AcPhkpg"

def call_mistral(prompt):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {MISTRAL_API_KEY}'
    }
    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"}
    }
    
    print("Sending data to Mistral API... (This may take a minute)")
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        print(f"Mistral API Error {response.status_code}: {response.text}")
        return None
        
    res_json = response.json()
    try:
        content = res_json['choices'][0]['message']['content']
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        return json.loads(content)
    except Exception as e:
        print("Failed to parse Mistral response:", e)
        return None


def extract_essential_data(lines):
    essential = []
    for line in lines:
        essential.append({
            "adlib": line.get("adlib", False),
            "beats": line.get("beats", 0),
            "notes": line.get("notes", []),
            "octaves": line.get("octaves", [])
        })
    return essential

import os

def build_instrument_profiles(samples_dir):
    profiles = {}
    print(f"\nScanning samples in {samples_dir} to build instrument profiles...")
    if not os.path.exists(samples_dir):
        print("Samples directory not found. Defaulting to harmonium.")
        return profiles
        
    for inst_folder in os.listdir(samples_dir):
        inst_path = os.path.join(samples_dir, inst_folder)
        if os.path.isdir(inst_path):
            mfccs = []
            files = [f for f in os.listdir(inst_path) if f.endswith('.mp3')]
            if not files:
                continue
            
            # Use up to 3 files to build an average profile
            for f in files[:3]:
                file_path = os.path.join(inst_path, f)
                try:
                    y_samp, sr_samp = librosa.load(file_path, sr=22050, mono=True)
                    mfcc = librosa.feature.mfcc(y=y_samp, sr=sr_samp, n_mfcc=13)
                    mfccs.append(np.mean(mfcc, axis=1))
                except Exception:
                    pass
            
            if mfccs:
                profiles[inst_folder.lower()] = np.mean(mfccs, axis=0)
                
    if profiles:
        print(f"Loaded profiles for: {', '.join(profiles.keys())}")
    return profiles

def classify_instrument(audio_chunk, sr, profiles, allowed_insts=None):
    if not profiles or len(audio_chunk) < sr * 0.1: # Need at least 100ms
        return "harmonium"
        
    try:
        # Resample chunk to match profile sample rate (22050)
        if sr != 22050:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=22050)
            
        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=22050, n_mfcc=13)
        chunk_profile = np.mean(mfcc, axis=1)
        
        best_inst = "harmonium"
        min_dist = float('inf')
        
        for inst, prof in profiles.items():
            if allowed_insts and inst not in allowed_insts:
                continue
            dist = np.linalg.norm(chunk_profile - prof) # Euclidean distance
            if dist < min_dist:
                min_dist = dist
                best_inst = inst
                
        return best_inst
    except Exception:
        return "harmonium"

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

def snap_to_scale(index, allowed_names):
    if not allowed_names:
        return index
    note_name = SWARA_NAMES[index % 12]
    if note_name in allowed_names:
        return index
    min_dist = 12
    best_idx = index
    for i in range(index - 12, index + 13):
        if SWARA_NAMES[i % 12] in allowed_names:
            dist = abs(i - index)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
    return best_idx

def get_swara_info(freq, sa_hz, allowed_notes=None):
    if freq < (sa_hz * 0.5): return "-", 0
    try:
        semitones = 12 * math.log2(freq / sa_hz)
        index = int(round(semitones))
        if allowed_notes:
            index = snap_to_scale(index, allowed_notes)
        return SWARA_NAMES[index % 12], index // 12
    except:
        return "-", 0

def process_beat_buffer(buffer):
    """
    Analyzes a buffer of detected pitches within one beat.
    Uses dynamic RLE and filters out transient noise (<15% of beat)
    to naturally extract the exact number of sub-beats (1, 2, 3, or 4).
    """
    if not buffer: return ["-"], [0]
    
    # 1. RLE grouping by Note Name
    rle = []
    curr_note = buffer[0][0]
    octs = [buffer[0][1]]
    count = 1
    
    from collections import Counter
    for n in buffer[1:]:
        if n[0] == curr_note:
            count += 1
            octs.append(n[1])
        else:
            common_oct = Counter(octs).most_common(1)[0][0]
            rle.append((curr_note, common_oct, count))
            curr_note = n[0]
            octs = [n[1]]
            count = 1
            
    common_oct = Counter(octs).most_common(1)[0][0]
    rle.append((curr_note, common_oct, count))
        
    # 2. Filter out transient notes (must be >= 15% of the total beat length)
    threshold = 0.15 * len(buffer)
    clean_rle = [g for g in rle if g[2] >= threshold]
    
    if not clean_rle:
        # If everything was filtered out, take the longest
        longest = max(rle, key=lambda x: x[2])
        clean_rle = [longest]
        
    # 3. Drop silences unless the beat is entirely silence
    final_notes = []
    final_octs = []
    
    for g in clean_rle:
        if g[0] != "-":
            final_notes.append(g[0])
            final_octs.append(g[1])
            
    if not final_notes:
        return ["-"], [0]
        
    return final_notes, final_octs

# --- MAIN RUNTIME ---
try:
    audio_file = input("Enter Audio Filename (e.g. song.wav): ").strip()
    tempo_factor_in = input("Enter Tempo Factor (e.g. 0.5 for half speed, 1 for normal): ").strip()
    tempo_factor = float(tempo_factor_in) if tempo_factor_in else 1.0
    
    print("Loading audio file... Please wait.")
    y, sr = librosa.load(audio_file, sr=44100, mono=True)
    max_dur_in = input("Enter max duration to process in seconds (press Enter for full song): ").strip()
    if max_dur_in:
        max_dur = float(max_dur_in)
        y = y[: int(max_dur * sr)]
        print(f"Audio trimmed to {max_dur} seconds of original audio.")
        
    if tempo_factor != 1.0:
        print(f"Time-stretching audio by factor {tempo_factor}...")
        y = librosa.effects.time_stretch(y, rate=tempo_factor)
        
    print("Audio loaded successfully.")
    
    instrument_profiles = build_instrument_profiles("public/samples")
    
    scale_in = input("Enter Scale (e.g. Pandhri 1, C4, or 261.6): ").strip()
    try:
        SA_HZ = float(scale_in)
    except ValueError:
        SA_HZ = SCALE_MAP.get(scale_in, 155.56)
        
    raag_notes_in = input("Enter allowed notes in Raag separated by space (e.g. Sa ga ma dha ni for Malkosh, or press Enter for all): ").strip()
    ALLOWED_NOTES = set(raag_notes_in.split()) if raag_notes_in else None
    
    adlib_inst_in = input("Enter expected Adlib instruments separated by comma (e.g. flute, sitar, or press Enter for all): ").strip().lower()
    ALLOWED_INSTS = [i.strip() for i in adlib_inst_in.split(',')] if adlib_inst_in else None
    
    voice_start_in = input("Enter start time of human voice in seconds (press Enter if none): ").strip()
    VOICE_START_SEC = float(voice_start_in) if voice_start_in else None
    
    song_bpm = int(input("Enter BPM of the Song (e.g. 160): "))
    target_bpm = song_bpm
    effective_song_bpm = song_bpm * tempo_factor
    record_bpm = song_bpm
    beats_in = input("Enter Taal Beats per Line (e.g. 8 for Keherwa, 7 for Rupak): ").strip()
    beats_per_line = int(beats_in) if beats_in else 8
    
    print("Analyzing audio to detect Adlib/Taal boundary...")
    import scipy.ndimage
    y_percussive = librosa.effects.hpss(y, margin=3.0)[1]
    rms_p = librosa.feature.rms(y=y_percussive)[0]
    rms_smoothed = scipy.ndimage.gaussian_filter1d(rms_p, sigma=int(sr/512))
    threshold = np.max(rms_smoothed) * 0.35
    taal_start_frame = np.argmax(rms_smoothed > threshold)
    taal_start_idx = librosa.frames_to_samples(taal_start_frame)
    if taal_start_idx > len(y) - sr:
        taal_start_idx = len(y) * 2
        print("Could not confidently detect Taal start. Spacebar override is active.")
    else:
        print(f"Detected TAAL start at {taal_start_idx/sr:.2f} seconds.")
except Exception as e:
    print(f"Error initializing: {e}")
    sys.exit(1)

CHUNK = 2048 # Increased for YIN algorithm stability
RATE = 44100
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

all_lines = []
curr_line_notes, curr_line_octs = [], []
last_actual_note, last_actual_oct = None, None
note_buffer = []
beat_dur = 60.0 / effective_song_bpm
mode = "ADLIB"
current_mode_start_idx = 0
last_beat_idx = -1
recording_started = False
adlib_silence_threshold = max(2, int(effective_song_bpm / 60.0))
adlib_silence_beats = 0

print(f"\n[Recording] Target BPM: {song_bpm} (Processing at {effective_song_bpm}) | Scale: {scale_in}")
print("Press SPACE for SAM/TAAL mode | ESC to Save & Exit")

current_line_start_idx = 0
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
        swara, octv = get_swara_info(freq, SA_HZ, ALLOWED_NOTES)
        note_buffer.append((swara, octv))
        
        elapsed_audio = (i - current_mode_start_idx) / sr
        total_beats = int(elapsed_audio / beat_dur)

        # Sync/Mode Switching
        trigger_taal = False
        if mode == "ADLIB" and i >= taal_start_idx:
            trigger_taal = True
            
        if msvcrt.kbhit():
            key = ord(msvcrt.getch())
            if key == 32 and mode == "ADLIB": # Space
                trigger_taal = True
                print(" (Manual Spacebar Trigger) ", end="")
            elif key == 27: # Esc
                break
                
        if trigger_taal:
            # Trim any trailing silence from the Adlib line before switching
            while curr_line_notes and curr_line_notes[-1] == ["-"]:
                curr_line_notes.pop()
                curr_line_octs.pop()
                
            if curr_line_notes and any(n != ["-"] for n in curr_line_notes):
                line_audio = y[current_line_start_idx : i + CHUNK]
                line_start_sec = current_line_start_idx / sr
                if VOICE_START_SEC is not None and line_start_sec >= VOICE_START_SEC:
                    predicted_inst = "harmonium"
                else:
                    predicted_inst = classify_instrument(line_audio, sr, instrument_profiles, ALLOWED_INSTS)
                all_lines.append({
                    "section": "",
                    "line_instrument": predicted_inst,
                    "percussion": "mute",
                    "line_volume": 1.0,
                    "adlib": True,
                    "legato": False,
                    "line_bpm": None,
                    "taal_key": str(beats_per_line),
                    "beats": len(curr_line_notes),
                    "lyrics": [[""] for _ in curr_line_notes],
                    "notes": curr_line_notes,
                    "octaves": curr_line_octs,
                    "meend": [[False] * len(b) for b in curr_line_notes]
                })
                curr_line_notes, curr_line_octs = [], []
                last_actual_note, last_actual_oct = None, None
            
            current_line_start_idx = i + CHUNK
            
            mode = "TAAL"
            current_mode_start_idx = i
            beat_dur = 60.0 / effective_song_bpm
            total_beats = 0
            last_beat_idx = -1
            print("\n*** SYNCED TO SAM (TAAL MODE) - ADLIB OVER ***")

        # Process a completed beat
        if total_beats != last_beat_idx:
            beat_notes, beat_octs = process_beat_buffer(note_buffer)
            
            if not recording_started:
                if beat_notes == ["-"]:
                    last_beat_idx = total_beats
                    note_buffer = []
                    continue
                recording_started = True

            print_notes = beat_notes
            
            if beat_notes != ["-"]:
                if last_actual_note is not None and beat_notes == last_actual_note and beat_octs == last_actual_oct:
                    beat_notes = ["~"]
                    beat_octs = [0]
                else:
                    last_actual_note = beat_notes
                    last_actual_oct = beat_octs
            else:
                last_actual_note = None
                last_actual_oct = None

            curr_line_notes.append(beat_notes)
            curr_line_octs.append(beat_octs)
            
            print(f"Beat {total_beats + 1}: {'-'.join(print_notes)}      ", end='\r')
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
                line_audio = y[current_line_start_idx : i + CHUNK]
                line_start_sec = current_line_start_idx / sr
                if VOICE_START_SEC is not None and line_start_sec >= VOICE_START_SEC:
                    predicted_inst = "harmonium"
                else:
                    predicted_inst = classify_instrument(line_audio, sr, instrument_profiles, ALLOWED_INSTS)
                all_lines.append({
                    "section": "",
                    "line_instrument": predicted_inst,
                    "percussion": "tabla" if mode == "TAAL" else "mute",
                    "line_volume": 1.0,
                    "adlib": mode == "ADLIB",
                    "legato": False,
                    "line_bpm": None,
                    "taal_key": str(beats_per_line),
                    "beats": len(curr_line_notes),
                    "lyrics": [[""] for _ in curr_line_notes],
                    "notes": curr_line_notes,
                    "octaves": curr_line_octs,
                    "meend": [[False] * len(b) for b in curr_line_notes]
                })
                curr_line_notes, curr_line_octs = [], []
                last_actual_note, last_actual_oct = None, None
                current_line_start_idx = i + CHUNK
            
            last_beat_idx = total_beats

    # Flush remaining notes at EOF
    if curr_line_notes and any(n != ["-"] for n in curr_line_notes):
        line_audio = y[current_line_start_idx :]
        line_start_sec = current_line_start_idx / sr
        if VOICE_START_SEC is not None and line_start_sec >= VOICE_START_SEC:
            predicted_inst = "harmonium"
        else:
            predicted_inst = classify_instrument(line_audio, sr, instrument_profiles, ALLOWED_INSTS)
        all_lines.append({
            "section": "",
            "line_instrument": predicted_inst,
            "percussion": "tabla" if mode == "TAAL" else "mute",
            "line_volume": 1.0,
            "adlib": mode == "ADLIB",
            "legato": mode == "ADLIB",
            "line_bpm": None,
            "taal_key": str(beats_per_line),
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
print("Saved raw mathematical data to precision_output.json")

# --- AI ENHANCEMENT PROMPT ---
use_ai = input("\nDo you want to use AI to automatically clean up this transcription? (y/n): ").strip().lower()
if use_ai == 'y':
    try:
        raw_lines = extract_essential_data(final_output.get("lines", []))
        
        prompt = f"""
You are an expert Indian Classical musician and transcriber. 
I have a system that automatically extracts raw vocal pitch data from audio into a JSON format. However, the raw data is mathematically too precise—it captures every micro-vibration, vibrato, and grace note.

Your task is to analyze the RAW TARGET DATA and clean it up using your musical intuition.

CRITICAL RULES:
1. Ignore minor pitch wobbles (vibrato/andolan). Group sustained notes into `~` ONLY IF they are across different beats. 
2. AGGRESSIVELY CONSOLIDATE SUB-BEATS: If a beat contains multiple sub-notes but they are functionally the same held note with vibrato or silences (e.g. `["Sa", "Sa", "re", "Sa"]`, `["Sa", "~", "~", "~"]`, or `["Sa", "-", "-", "-"]`), smooth them out into a SINGLE full beat: `["Sa"]`.
3. Long Adlib sections MUST be split into multiple separate lines if you detect distinct musical phrases separated by silence (`-`).
4. Ensure the total number of beats across your split Adlib lines equals the total number of beats in the original Adlib line.
5. Do NOT change the number of lines or beats for TAAL (non-adlib) sections. The TAAL sections strictly follow a {beats_per_line}-beat structure. Do not add or remove beats.
6. If a note is `~`, its octave MUST be `[0]`.
7. Preserve sub-beat arrays (e.g., `["Pa", "Ni", "Dha"]`) ONLY if they represent a genuine fast musical phrase (triplet/quarter). DO NOT output arrays like `["Sa", "~", "~", "~"]` for a single sustained note; just use `["Sa"]`.

RAW TARGET DATA:
{json.dumps(raw_lines, indent=2)}

Return ONLY a valid JSON object with a single key "cleaned_lines" containing the array of objects representing the cleaned lines. It must match the structure of the input EXACTLY. No markdown formatting.
"""
        cleaned_essential_lines = call_gemini(prompt)
        if not cleaned_essential_lines:
            print("\nGemini failed. Falling back to Mistral API...")
            cleaned_essential_lines = call_mistral(prompt)
        
        if cleaned_essential_lines:
            if isinstance(cleaned_essential_lines, dict) and "cleaned_lines" in cleaned_essential_lines:
                cleaned_essential_lines = cleaned_essential_lines["cleaned_lines"]
            
            new_lines = []
            for clean_line in cleaned_essential_lines:
                is_adlib = clean_line.get("adlib", False)
                new_lines.append({
                    "section": "",
                    "line_instrument": "harmonium",
                    "percussion": "mute" if is_adlib else "tabla",
                    "line_volume": 1.0,
                    "adlib": is_adlib,
                    "legato": is_adlib,
                    "line_bpm": None,
                    "taal_key": str(beats_per_line),
                    "beats": clean_line.get("beats", len(clean_line.get("notes", []))),
                    "lyrics": [[""] for _ in clean_line.get("notes", [])],
                    "notes": clean_line.get("notes", []),
                    "octaves": clean_line.get("octaves", []),
                    "meend": [[False] * len(b) for b in clean_line.get("notes", [])]
                })
                
            final_output["lines"] = new_lines
            with open("swar_laya_output.json", 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2)
            print("\nSuccessfully generated AI-cleaned transcription: swar_laya_output.json")
        else:
            print("AI transcription failed.")
            
    except Exception as e:
        print(f"Error reading reference file or generating AI transcription: {e}")