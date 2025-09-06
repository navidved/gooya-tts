import os
import re
import random
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
MIN_DUR = 1.0   # حداقل 1 ثانیه
MAX_DUR = 15.0  # حداکثر 15 ثانیه

# ایجاد دایرکتوری‌ها
os.makedirs(OUT_WAV_DIR, exist_ok=True)
os.makedirs(OUT_FL_DIR, exist_ok=True)

# نگاشت کاراکترهای عربی به فارسی
ARABIC_TO_FA = {
    '\u064A': 'ی', '\u0643': 'ک', '\u06C0': 'ه', 
    '\u06CC': 'ی', '\u0649': 'ی', '\u06A9': 'ک'
}

# تبدیل اعداد فارسی به انگلیسی
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
LATIN_DIGITS = "0123456789"
DIGIT_MAP = {ord(p): LATIN_DIGITS[i] for i, p in enumerate(PERSIAN_DIGITS)}

# حرکات عربی
DIACRITICS = ''.join(chr(c) for c in range(0x064B, 0x065F+1)) + "\u0670"
DIAC_RE = re.compile(f"[{re.escape(DIACRITICS)}]")

def normalize_persian_text(text):
    """نرمال‌سازی متن فارسی"""
    # حذف کاراکترهای کنترلی
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    
    # تبدیل کاراکترهای عربی
    text = ''.join(ARABIC_TO_FA.get(ch, ch) for ch in text)
    
    # حذف حرکات
    text = DIAC_RE.sub("", text)
    
    # تبدیل اعداد فارسی به انگلیسی
    text = text.translate(DIGIT_MAP)
    
    # حذف تطویل
    text = text.replace("\u0640", "")
    
    # تبدیل نیم‌فاصله به فاصله
    text = text.replace("\u200c", " ")
    
    # نرمال‌سازی فاصله‌ها
    text = re.sub(r"\s+", " ", text).strip()
    
    # تبدیل گیومه فارسی
    text = re.sub(r"[«»]", "\"", text)
    
    # حذف کاراکترهای غیرضروری
    text = re.sub(r'[^\u0600-\u06FF\u0020-\u007E\s]', '', text)
    
    return text.strip()

def process_audio(audio_path, target_sr=22050):
    """بارگذاری و پردازش صوتی"""
    try:
        # بارگذاری با librosa
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # حذف سکوت
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # نرمال‌سازی
        peak = np.abs(audio).max()
        if peak > 0:
            audio = (0.95 / peak) * audio
        
        # حذف DC offset
        audio = audio - np.mean(audio)
        
        return audio, target_sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None, None

def main():
    print("🔄 Loading dataset from HuggingFace...")
    
    # لود دیتاست بدون Audio casting
    ds = load_dataset(HF_DATASET, split="train")
    
    print(f"📊 Total samples in dataset: {len(ds)}")
    
    samples = []
    skipped = 0
    
    print("🎵 Processing audio samples...")
    for idx, item in enumerate(tqdm(ds, desc="Processing")):
        try:
            # نرمال‌سازی متن
            text = normalize_persian_text(item["sentence"])
            
            # بررسی طول متن
            if len(text) < 5 or len(text) > 500:
                skipped += 1
                continue
            
            # مسیر فایل صوتی
            # فرض می‌کنیم audio field حاوی path یا bytes است
            audio_data = item["audio"]
            
            # اگر audio یک dictionary است با array و sampling_rate
            if isinstance(audio_data, dict) and 'array' in audio_data:
                audio_array = np.array(audio_data['array'])
                original_sr = audio_data.get('sampling_rate', SR)
                
                # resample اگر نیاز است
                if original_sr != SR:
                    audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=SR)
                
            # اگر audio یک path است
            elif isinstance(audio_data, str):
                audio_array, _ = process_audio(audio_data, SR)
                if audio_array is None:
                    skipped += 1
                    continue
            
            # اگر audio یک bytes object است
            elif isinstance(audio_data, bytes):
                # ذخیره موقت و بارگذاری
                temp_path = f"/tmp/temp_audio_{idx}.wav"
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                audio_array, _ = process_audio(temp_path, SR)
                os.remove(temp_path)
                if audio_array is None:
                    skipped += 1
                    continue
            else:
                # تلاش برای تبدیل مستقیم
                try:
                    audio_array = np.array(audio_data)
                except:
                    print(f"Unknown audio format for item {idx}")
                    skipped += 1
                    continue
            
            # پردازش صدا
            # حذف سکوت
            audio_array, _ = librosa.effects.trim(audio_array, top_db=20)
            
            # نرمال‌سازی
            peak = np.abs(audio_array).max()
            if peak > 0:
                audio_array = (0.95 / peak) * audio_array
            
            # بررسی مدت زمان
            duration = len(audio_array) / SR
            if duration < MIN_DUR or duration > MAX_DUR:
                skipped += 1
                continue
            
            # ذخیره فایل WAV
            filename = f"sample_{idx:06d}.wav"
            filepath = os.path.join(OUT_WAV_DIR, filename)
            sf.write(filepath, audio_array, SR, subtype="PCM_16")
            
            samples.append((filepath, text, duration))
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            skipped += 1
            continue
    
    print(f"\n📊 Processing complete:")
    print(f"  - Valid samples: {len(samples)}")
    print(f"  - Skipped samples: {skipped}")
    
    if len(samples) == 0:
        print("❌ No valid samples found!")
        return
    
    # آمار
    total_duration = sum(s[2] for s in samples)
    print(f"  - Total duration: {total_duration/3600:.2f} hours")
    print(f"  - Average duration: {total_duration/len(samples):.2f} seconds")
    
    # تقسیم داده‌ها (95% train, 5% validation)
    random.seed(42)
    random.shuffle(samples)
    
    n_val = max(300, int(0.05 * len(samples)))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    # ذخیره فایل‌لیست‌ها
    print("\n📝 Writing filelists...")
    
    with open(os.path.join(OUT_FL_DIR, "train.txt"), "w", encoding="utf-8") as f:
        for filepath, text, _ in train_samples:
            f.write(f"{filepath}|{text}\n")
    
    with open(os.path.join(OUT_FL_DIR, "val.txt"), "w", encoding="utf-8") as f:
        for filepath, text, _ in val_samples:
            f.write(f"{filepath}|{text}\n")
    
    print(f"✅ Dataset preparation complete!")
    print(f"  - Train samples: {len(train_samples)}")
    print(f"  - Validation samples: {len(val_samples)}")

if __name__ == "__main__":
    main()