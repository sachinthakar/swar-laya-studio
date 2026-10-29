import sys
import json
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
            "responseMimeType": "application/json"
        }
    }
    
    print("Sending data to Gemini API... (This may take a minute)")
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
        sys.exit(1)
        
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
        print("Raw response:", res_json)
        sys.exit(1)

def extract_essential_data(lines):
    # Strip heavy arrays (lyrics, meend) to save tokens
    essential = []
    for line in lines:
        essential.append({
            "adlib": line.get("adlib", False),
            "beats": line.get("beats", 0),
            "notes": line.get("notes", []),
            "octaves": line.get("octaves", [])
        })
    return essential

def main():
    if len(sys.argv) != 4:
        print("Usage: python ai_transcriber.py <raw_target.json> <manual_reference.json> <output.json>")
        sys.exit(1)
        
    raw_file = sys.argv[1]
    ref_file = sys.argv[2]
    out_file = sys.argv[3]
    
    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    with open(ref_file, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
        
    raw_lines = extract_essential_data(raw_data.get("lines", []))
    ref_lines = extract_essential_data(ref_data.get("lines", []))
    
    prompt = f"""
You are an expert Indian Classical musician and transcriber. 
I have a system that automatically extracts raw vocal pitch data from audio into a JSON format. However, the raw data is mathematically too precise—it captures every micro-vibration, vibrato, and grace note, resulting in messy, robotic-sounding notes that look like `[["Sa"], ["Sa"], ["re"], ["Sa"]]`.

When a human transcribes the same audio, they intuitively filter out the noise and group the notes into clean musical phrases, utilizing the sustain character `~` when a note is held, e.g., `[["Sa"], ["~"], ["~"], ["~"]]`.

Here is a REFERENCE example of how a human manually transcribes a song beautifully:
{json.dumps(ref_lines, indent=2)}

Now, I will provide you with the RAW pitch data for a NEW song.
Your task is to analyze the RAW notes and octaves, and clean them up using the exact same stylistic intuition shown in the reference above. 
- Ignore minor pitch wobbles (vibrato/andolan).
- Group sustained notes into `~`.
- Form logical musical phrases appropriate for Indian classical/semi-classical music.
- **CRITICAL**: The raw data may have clumped a very long Adlib section into a single massive line. You MUST split long Adlib sections into multiple separate lines if you detect distinct musical phrases separated by silence (`-`). This is necessary so the user can assign different instruments (e.g. Flute, then Sitar) to different Adlib lines.
- Ensure the total number of beats across your split Adlib lines equals the total number of beats in the original Adlib line.
- Do NOT change the number of lines for TAAL (non-adlib) sections.
- If a note is `~`, its octave should be `[0]`.

RAW TARGET DATA:
{json.dumps(raw_lines, indent=2)}

Return ONLY a valid JSON array of objects representing the cleaned lines. It must match the structure of the input EXACTLY:
[
  {{ "adlib": true, "beats": 4, "notes": [["Sa"], ["~"], ["~"], ["Re"]], "octaves": [[0], [0], [0], [0]] }}, ...
]
"""

    cleaned_essential_lines = call_gemini(prompt)
    
    # Reconstruct the full output JSON by building a new lines array
    final_output = copy.deepcopy(raw_data)
    new_lines = []
    
    # We need to extract the adlib_bpm from the raw_data if it exists (usually stored in the first adlib line's line_bpm)
    adlib_bpm = None
    for l in raw_data.get("lines", []):
        if l.get("adlib") and l.get("line_bpm"):
            adlib_bpm = l.get("line_bpm")
            break
            
    for clean_line in cleaned_essential_lines:
        is_adlib = clean_line.get("adlib", False)
        new_lines.append({
            "section": "",
            "line_instrument": "harmonium",
            "percussion": "mute" if is_adlib else "tabla",
            "line_volume": 1.0,
            "adlib": is_adlib,
            "legato": is_adlib,
            "line_bpm": adlib_bpm if is_adlib else None,
            "taal_key": "",
            "beats": clean_line.get("beats", len(clean_line.get("notes", []))),
            "lyrics": [[""] for _ in clean_line.get("notes", [])],
            "notes": clean_line.get("notes", []),
            "octaves": clean_line.get("octaves", []),
            "meend": [[False] * len(b) for b in clean_line.get("notes", [])]
        })
        
    final_output["lines"] = new_lines
            
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Successfully generated cleaned transcription: {out_file}")

if __name__ == "__main__":
    main()
