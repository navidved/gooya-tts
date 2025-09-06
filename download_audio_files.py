import os
import re
import random
import unicodedata
import soundfile as sf
import librosa
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# تنظیمات
HF_DATASET = "navidved/approved-tts-dataset"
OUT_WAV_DIR = "data_fa/wavs"
OUT_FL_DIR = "filelists"
SR = 22050
MIN_DUR = 1.0
MAX_DUR = 15.0

os.makedirs(OUT_WAV_DIR, exist_ok=True)
os.makedirs(OUT_FL_DIR, exist_ok=True)

# Persian text normalization
ARABIC_TO_FA = {
    '\u064A': 'ی', '\u0643': 'ک', '\u06C0': 'ه',
    '\u06CC': 'ی', '\u0649': 'ی', '\u06A9': 'ک'
}

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
LATIN_DIGITS = "0123456789"
DIGIT_MAP = {ord(p): LATIN_DIGITS[i] for i, p in enumerate(PERSIAN_DIGITS)}

DIACRITICS = ''.join(chr(c) for c in range(0x064B, 0x065F+1)) + "\u0670"
DIAC_RE = re.compile(f"[{re.escape(DIACRITICS)}]")

def normalize_persian_text(text):
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = ''.join(ARABIC_TO_FA.get(ch, ch) for ch in text)
    text = DIAC_RE.sub("", text)
    text = text.translate(DIGIT_MAP)
    text = text.replace("\u0640", "")
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[«»]", "\"", text)
    text = re.sub(r'[^\u0600-\u06FF\u0020-\u007E\s]', '', text)
    return text.strip()

def main():
    print("🔄 Step 1: Loading metadata from HuggingFace...")
    
    # لود metadata بدون audio
    ds_metadata = load_dataset(
        HF_DATASET,
        split="train",
        use_auth_token=True
    )
    
    print(f"📊 Total samples: {len(ds_metadata)}")
    
    # لیست فایل‌های موجود در repo
    print("\n🔍 Step 2: Checking available audio files in repo...")
    try:
        files = list_repo_files(HF_DATASET, repo_type="dataset")
        audio_files = [f for f in files if f.endswith(('.mp3', '.wav', '.flac'))]
        print(f"Found {len(audio_files)} audio files in repository")
    except Exception as e:
        print(f"Could not list repo files: {e}")
        audio_files = []
    
    samples = []
    cache_dir = "audio_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    print("\n🎵 Step 3: Processing samples...")
    
    for idx in tqdm(range(min(100, len(ds_metadata))), desc="Processing"):  # فقط 100 نمونه اول برای تست
        try:
            item = ds_metadata[idx]
            
            # نرمال‌سازی متن
            text = normalize_persian_text(item["sentence"])
            
            if len(text) < 5 or len(text) > 500:
                continue
            
            # پیدا کردن فایل صوتی مرتبط
            file_name = item.get("file_name", "")
            
            if file_name:
                # تلاش برای دانلود فایل از repo
                try:
                    local_path = hf_hub_download(
                        repo_id=HF_DATASET,
                        filename=file_name,
                        repo_type="dataset",
                        cache_dir=cache_dir,
                        use_auth_token=True
                    )
                    
                    # تبدیل به WAV
                    audio, sr_orig = librosa.load(local_path, sr=SR, mono=True)
                    
                    # پردازش
                    audio, _ = librosa.effects.trim(audio, top_db=20)
                    peak = np.abs(audio).max()
                    if peak > 0:
                        audio = (0.95 / peak) * audio
                    
                    duration = len(audio) / SR
                    if duration < MIN_DUR or duration > MAX_DUR:
                        continue
                    
                    # ذخیره
                    output_filename = f"sample_{idx:06d}.wav"
                    output_path = os.path.join(OUT_WAV_DIR, output_filename)
                    sf.write(output_path, audio, SR, subtype="PCM_16")
                    
                    samples.append((output_path, text))
                    
                except Exception as e:
                    print(f"\nCould not download {file_name}: {e}")
                    continue
            
        except Exception as e:
            print(f"\nError processing item {idx}: {e}")
            continue
    
    print(f"\n📊 Processing complete: {len(samples)} valid samples")
    
    if len(samples) == 0:
        print("❌ No samples processed successfully!")
        print("\n💡 Try manual download:")
        print("1. Go to: https://huggingface.co/datasets/navidved/approved-tts-dataset")
        print("2. Download the dataset files manually")
        print("3. Extract and process locally")
        return
    
    # تقسیم و ذخیره
    random.seed(42)
    random.shuffle(samples)
    
    n_val = max(10, int(0.05 * len(samples)))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    with open(os.path.join(OUT_FL_DIR, "train.txt"), "w", encoding="utf-8") as f:
        for filepath, text in train_samples:
            f.write(f"{filepath}|{text}\n")
    
    with open(os.path.join(OUT_FL_DIR, "val.txt"), "w", encoding="utf-8") as f:
        for filepath, text in val_samples:
            f.write(f"{filepath}|{text}\n")
    
    print(f"✅ Complete! Train: {len(train_samples)}, Val: {len(val_samples)}")

if __name__ == "__main__":
    main()