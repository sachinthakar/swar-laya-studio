import librosa
import numpy as np
import os

def build_instrument_profiles(samples_dir):
    profiles = {}
    print(f"Scanning samples in {samples_dir}...")
    if not os.path.exists(samples_dir):
        print("Directory does not exist.")
        return profiles
        
    for inst_folder in os.listdir(samples_dir):
        inst_path = os.path.join(samples_dir, inst_folder)
        if os.path.isdir(inst_path):
            mfccs = []
            files = [f for f in os.listdir(inst_path) if f.endswith('.mp3')]
            if not files:
                continue
            
            # Use only first 2 files to keep it fast
            for f in files[:2]:
                file_path = os.path.join(inst_path, f)
                try:
                    y, sr = librosa.load(file_path, sr=22050, mono=True)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfccs.append(np.mean(mfcc, axis=1))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            if mfccs:
                profiles[inst_folder.lower()] = np.mean(mfccs, axis=0)
                print(f"Loaded profile for {inst_folder.lower()}")
                
    return profiles

if __name__ == "__main__":
    profiles = build_instrument_profiles("c:/Personal/Swar-Laya Studio/public/samples")
    print("Profiles:", profiles.keys())
    for inst, prof in profiles.items():
        print(f"{inst} shape: {prof.shape}")
