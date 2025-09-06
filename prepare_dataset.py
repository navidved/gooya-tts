import os
import re
import random
from pathlib import Path
import unicodedata
import soundfile as sf
import librosa
import numpy as np
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

# Persian normalization
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
    """نرمال‌سازی متن فارسی"""
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
    print("🔄 Loading dataset from HuggingFace...")
    
    # لود دیتاست (از کش استفاده می‌کند اگر قبلاً دانلود شده)
    ds = load_dataset(HF_DATASET, split="train", use_auth_token=True)
    
    print(f"📊 Total samples in dataset: {len(ds)}")
    
    samples = []
    skipped_short = 0
    skipped_long = 0
    skipped_text = 0
    
    print("\n🎵 Processing all audio samples...")
    
    for idx in tqdm(range(len(ds)), desc="Processing"):
        try:
            item = ds[idx]
            
            # نرمال‌سازی متن
            text = normalize_persian_text(item["sentence"])
            
            if len(text) < 5 or len(text) > 500:
                skipped_text += 1
                continue
            
            # دریافت داده‌های صوتی
            audio_data = item['audio']
            
            # استخراج array و sampling_rate
            audio_array = np.array(audio_data['array'], dtype=np.float32)
            original_sr = audio_data['sampling_rate']
            
            # Resample به 22050 Hz اگر نیاز است
            if original_sr != SR:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=original_sr, 
                    target_sr=SR
                )
            
            # حذف سکوت
            audio_array, _ = librosa.effects.trim(audio_array, top_db=20)
            
            # نرمال‌سازی سطح صدا
            peak = np.abs(audio_array).max()
            if peak > 0:
                audio_array = (0.95 / peak) * audio_array
            
            # حذف DC offset
            audio_array = audio_array - np.mean(audio_array)
            
            # بررسی مدت زمان
            duration = len(audio_array) / SR
            
            if duration < MIN_DUR:
                skipped_short += 1
                continue
            elif duration > MAX_DUR:
                skipped_long += 1
                continue
            
            # ذخیره فایل WAV
            filename = f"sample_{idx:06d}.wav"
            filepath = str(Path.cwd() / OUT_WAV_DIR / filename)
            
            # ذخیره با کیفیت 16-bit PCM
            sf.write(filepath, audio_array, SR, subtype="PCM_16")
            
            samples.append({
                'path': filepath,
                'text': text,
                'duration': duration,
                'speaker': item.get('speaker', 'default')
            })
            
        except Exception as e:
            print(f"\nError processing item {idx}: {e}")
            continue
    
    print(f"\n📊 Processing complete:")
    print(f"  - Valid samples: {len(samples)}")
    print(f"  - Skipped (short): {skipped_short}")
    print(f"  - Skipped (long): {skipped_long}")
    print(f"  - Skipped (text): {skipped_text}")
    
    if len(samples) == 0:
        print("❌ No valid samples found!")
        return
    
    # محاسبه آمار
    total_duration = sum(s['duration'] for s in samples)
    avg_duration = total_duration / len(samples)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  - Total duration: {total_duration/3600:.2f} hours")
    print(f"  - Average duration: {avg_duration:.2f} seconds")
    print(f"  - Min duration: {min(s['duration'] for s in samples):.2f} seconds")
    print(f"  - Max duration: {max(s['duration'] for s in samples):.2f} seconds")
    
    # تقسیم داده‌ها (95% train, 5% validation)
    random.seed(42)
    random.shuffle(samples)
    
    n_val = max(300, int(0.05 * len(samples)))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    # ذخیره فایل‌لیست‌ها
    print("\n📝 Writing filelists...")
    
    # Train filelist
    train_file = os.path.join(OUT_FL_DIR, "train.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(f"{sample['path']}|{sample['text']}\n")
    
    # Validation filelist
    val_file = os.path.join(OUT_FL_DIR, "val.txt")
    with open(val_file, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(f"{sample['path']}|{sample['text']}\n")
    
    print(f"✅ Dataset preparation complete!")
    print(f"  - Train samples: {len(train_samples)} → {train_file}")
    print(f"  - Validation samples: {len(val_samples)} → {val_file}")
    
    # ذخیره metadata
    metadata_file = os.path.join(OUT_FL_DIR, "metadata.txt")
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Train samples: {len(train_samples)}\n")
        f.write(f"Validation samples: {len(val_samples)}\n")
        f.write(f"Total duration: {total_duration/3600:.2f} hours\n")
        f.write(f"Sample rate: {SR} Hz\n")
        f.write(f"Min duration: {MIN_DUR} seconds\n")
        f.write(f"Max duration: {MAX_DUR} seconds\n")
    
    print(f"\n📊 Metadata saved to: {metadata_file}")
    
    # نمایش چند نمونه
    print("\n📝 Sample entries:")
    for i in range(min(3, len(train_samples))):
        sample = train_samples[i]
        print(f"  {i+1}. {sample['path']}|{sample['text'][:50]}...")

if __name__ == "__main__":
    main()