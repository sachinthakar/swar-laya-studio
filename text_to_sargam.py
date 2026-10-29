import sys
import json
import requests
import os

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
        "response_format": {"type": "json_object"}
    }
    
    print("Sending data to Mistral API... (This may take a minute)")
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
        sys.exit(1)
        
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
        print("Raw response:", res_json)
        sys.exit(1)

def main():
    print("=== Text to Sargam Mapper ===")
    file_name = input("Enter the notation file name (e.g., 'Zindagi milke bitayenge.txt'): ").strip()
    scale = input("Enter the Scale (e.g., 138.59 or 'Kali 1'): ").strip()
    if not scale:
        scale = "138.59"
    
    bpm_input = input("Enter the Song BPM (e.g., 160): ").strip()
    try:
        bpm = int(bpm_input)
    except ValueError:
        bpm = 160
        
    beats_input = input("Enter Taal Beats per Line (e.g., 8 for Keherwa): ").strip()
    beats = beats_input if beats_input else "8"
    
    notation_path = os.path.join("Input Notations", file_name)
    rules_path = os.path.join("Input Notations", "How to read SARGAM.txt")
    
    if not os.path.exists(notation_path):
        if os.path.exists(os.path.join("Input Notations", file_name + ".txt")):
            notation_path = os.path.join("Input Notations", file_name + ".txt")
        else:
            print(f"Error: Could not find notation file at {notation_path}")
            sys.exit(1)
            
    if not os.path.exists(rules_path):
        print(f"Error: Could not find rules file at {rules_path}")
        sys.exit(1)
        
    with open(notation_path, 'r', encoding='utf-8') as f:
        notation_text = f.read()
        
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules_text = f.read()
        
    prompt = f"""
You are an expert Indian Classical musician and transcriber.
Your task is to parse a free-form English text file containing Sargam notations and convert it into a highly structured JSON array of lines that is compatible with Swar-Laya Studio.

=== RULES FOR READING SARGAM ===
{rules_text}

=== INPUT NOTATION ===
{notation_text}

=== CONFIGURATION ===
Target Beats per Line for Taal sections: {beats}
Target BPM: {bpm}

=== OUTPUT INSTRUCTIONS ===
1. You must output ONLY a valid JSON object with the following structure:
{{
  "title": "{file_name.replace('.txt', '')}",
  "scale": "{scale}",
  "bpm": {bpm},
  "pre_roll": 0,
  "lines": [
     // array of line objects
  ]
}}

2. Structure of each line object in the "lines" array:
{{
  "section": "String (e.g. Mukhda, Antara 1, Prelude. Infer from text)",
  "line_instrument": "String (Infer from text. Fallback to 'harmonium' if not specified or available. Available: harmonium, flute, sitar, violin, piano)",
  "percussion": "String ('tabla' for Taal sections, 'mute' for Adlib sections)",
  "line_volume": 1,
  "adlib": boolean (true if it's an adlib section like Prelude/Interlude with no fixed taal structure, false if it's a song section with lyrics or fixed beats like Mukhda/Antara),
  "legato": true,
  "line_bpm": integer or null (Set to {bpm} if adlib is true, otherwise null),
  "taal_key": "String ('0' if adlib is true, otherwise e.g. '16' for Teentaal, '8' for Keherwa, '7' for Rupak. Usually '0' for adlib and '{beats}' for Taal)",
  "beats": "String (Number of beats. '{beats}' for Taal sections. For adlib, count the total notes/rests in the line)",
  "lyrics": [["Word1"], ["Word2"], ...],  // Array of arrays. One inner array per beat. Should match the notes length.
  "notes": [["Sa"], ["re", "Ga"], ["~"], ...], // Array of arrays. One inner array per beat. Divide notes in a beat equally.
  "octaves": [[0], [0, 1], [0], ...], // Array of arrays matching notes. 0=Middle, 1=High, 2=Low. For '~', use 0.
  "meend": [[false], [false, false], [true], ...] // Array of arrays matching notes. true if the note is part of a meend.
}}

3. Guidelines for Parsing:
- Map Capital letters to Shuddh notes (Sa, Re, Ga, Ma, Pa, Dha, Ni).
- Map small letters to Komal notes (re, ga, ma, dha, ni).
- Map M# or M^ to Tivra Ma (Ma). Note: standard Swar-Laya studio format usually expects Shuddh and Komal names, ensure to output valid names like "Sa", "re", "Re", "ga", "Ga", "Ma", "ma", "Pa", "dha", "Dha", "ni", "Ni".
- Low octave notes (,N or .N) -> octave 2.
- High octave notes (N') -> octave 1.
- Middle octave notes -> octave 0.
- A dash '-' represents a rest/silence.
- A tilde '~' represents a meend (glide/sustain). Set meend=true for these and previous notes if applicable, but output the note as "~" with octave 0 and meend false in standard Swar-Laya unless it's specifically a meend array flag. Just output the note name as "~" and octave as 0.
- Dots '...' mean sustain. Represent sustains with "~".
- Lyrics should be aligned with the notes. If the text has lines of lyrics followed by lines of notes, align the lyric syllables to the first note of the corresponding beats. For empty lyric beats, use "".
- Group notes separated by spaces into separate beats. Notes clustered together (like "PMP") might belong to the same beat depending on the pipe "|" separators.
- Use the pipe "|" to understand beat boundaries. If the text has `P | P M | R M | D PM P`, each pipe-separated section is one beat. So `notes` for that line would be `[["Pa"], ["Pa", "Ma"], ["Re", "Ma"], ["Dha", "Pa", "Ma", "Pa"]]`.
- Please do your absolute best to map the English text precisely to the Swar-Laya Studio JSON format.

Return ONLY the raw JSON output. No markdown wrappers if possible, but if you do, ensure it's strictly valid JSON.
"""

    output_json = call_mistral(prompt)
    
    out_dir = "output_jsons"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    base_name = file_name.replace('.txt', '') if file_name.endswith('.txt') else file_name
    out_path = os.path.join(out_dir, f"{base_name}_output.json")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2)
        
    print(f"\nSuccess! Saved Swar-Laya JSON to {out_path}")

if __name__ == "__main__":
    main()
