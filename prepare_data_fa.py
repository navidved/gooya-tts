import os
import re
import random
import unicodedata
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
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
    
    # حذف کاراکترهای غیرضروری (فقط فارسی، انگلیسی و علائم نگارشی)
    text = re.sub(r'[^\u0600-\u06FF\u0020-\u007E\s]', '', text)
    
    return text.strip()

def process_audio(audio, sr):
    """پردازش صوتی"""
    # حذف سکوت
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # نرمال‌سازی
    peak = np.abs(audio).max()
    if peak > 0:
        audio = (0.95 / peak) * audio
    
    # حذف DC offset
    audio = audio - np.mean(audio)
    
    return audio

def process_single_item(args):
    """پردازش یک نمونه"""
    idx, item, sr_target = args
    
    try:
        # نرمال‌سازی متن
        text = normalize_persian_text(item["sentence"])
        
        # بررسی طول متن
        if len(text) < 5 or len(text) > 500:
            return None
        
        # پردازش صدا
        audio = item["audio"]["array"]
        audio = process_audio(audio, item["audio"]["sampling_rate"])
        
        # بررسی مدت زمان
        duration = len(audio) / sr_target
        if duration < MIN_DUR or duration > MAX_DUR:
            return None
        
        # ذخیره فایل
        filename = f"sample_{idx:06d}.wav"
        filepath = os.path.join(OUT_WAV_DIR, filename)
        sf.write(filepath, audio, sr_target, subtype="PCM_16")
        
        return (filepath, text, duration)
    
    except Exception as e:
        print(f"Error processing item {idx}: {e}")
        return None

def main():
    print("🔄 Loading dataset from HuggingFace...")
    ds = load_dataset(HF_DATASET, split="train", use_auth_token=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    
    print(f"📊 Total samples in dataset: {len(ds)}")
    
    # پردازش موازی
    print("🎵 Processing audio samples...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        args = [(idx, item, SR) for idx, item in enumerate(ds)]
        results = list(tqdm(
            executor.map(process_single_item, args),
            total=len(ds),
            desc="Processing"
        ))
    
    # فیلتر نتایج
    samples = [r for r in results if r is not None]
    
    # آمار
    total_duration = sum(s[2] for s in samples)
    print(f"\n📈 Statistics:")
    print(f"  - Valid samples: {len(samples)}/{len(ds)}")
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