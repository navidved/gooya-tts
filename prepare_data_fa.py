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
MIN_DUR = 1.0
MAX_DUR = 15.0

os.makedirs(OUT_WAV_DIR, exist_ok=True)
os.makedirs(OUT_FL_DIR, exist_ok=True)

# نگاشت کاراکترها
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
    
    # لود بدون تبدیل Audio
    from datasets import Features, Value
    features = Features({
        'audio': Value('string'),  # یا هر type دیگری غیر از Audio
        'file_name': Value('string'),
        'sentence': Value('string'),
        'speaker': Value('string'),
        'duration': Value('float32'),
        'sample_rate': Value('int32')
    })
    
    try:
        # تلاش برای لود با features مشخص
        ds = load_dataset(
            HF_DATASET, 
            split="train",
            features=features,
            use_auth_token=True
        )
    except:
        # اگر نشد، بدون features
        ds = load_dataset(HF_DATASET, split="train", use_auth_token=True)
    
    print(f"📊 Total samples in dataset: {len(ds)}")
    
    # ابتدا فقط metadata را بررسی کنیم
    print("\n🔍 Checking dataset structure...")
    sample = ds[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Audio field type: {type(sample.get('audio'))}")
    
    samples = []
    skipped = 0
    
    print("\n🎵 Processing samples...")
    
    for idx in tqdm(range(len(ds)), desc="Processing"):
        try:
            item = ds[idx]
            
            # نرمال‌سازی متن
            text = normalize_persian_text(item["sentence"])
            
            if len(text) < 5 or len(text) > 500:
                skipped += 1
                continue
            
            # استخراج نام فایل
            file_name = item.get("file_name", f"sample_{idx}.mp3")
            
            # چک کردن duration
            duration = item.get("duration", 0)
            if duration < MIN_DUR or duration > MAX_DUR:
                skipped += 1
                continue
            
            # ذخیره اطلاعات (فعلاً بدون پردازش صوت)
            samples.append({
                'idx': idx,
                'text': text,
                'file_name': file_name,
                'duration': duration
            })
            
        except Exception as e:
            print(f"\nError processing item {idx}: {e}")
            skipped += 1
            continue
    
    print(f"\n📊 Initial processing complete:")
    print(f"  - Valid samples: {len(samples)}")
    print(f"  - Skipped samples: {skipped}")
    
    if len(samples) == 0:
        print("❌ No valid samples found!")
        return
    
    # حالا سعی کنیم فایل‌های صوتی را دانلود/پردازش کنیم
    print("\n📥 Processing audio files...")
    
    processed_samples = []
    
    for sample_info in tqdm(samples[:10], desc="Testing audio"):  # فقط 10 نمونه برای تست
        idx = sample_info['idx']
        
        try:
            # دریافت داده صوتی
            item = ds[idx]
            audio_data = item.get('audio')
            
            # اگر audio یک path است
            if isinstance(audio_data, str):
                print(f"Audio is path: {audio_data}")
                # TODO: دانلود یا بارگذاری از path
                
            # اگر audio یک bytes است
            elif isinstance(audio_data, bytes):
                print(f"Audio is bytes (size: {len(audio_data)} bytes)")
                # ذخیره موقت و پردازش
                temp_path = f"/tmp/temp_audio_{idx}.mp3"
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                
                # تبدیل به WAV
                audio, sr = librosa.load(temp_path, sr=SR, mono=True)
                os.remove(temp_path)
                
                # پردازش
                audio, _ = librosa.effects.trim(audio, top_db=20)
                peak = np.abs(audio).max()
                if peak > 0:
                    audio = (0.95 / peak) * audio
                
                # ذخیره
                filename = f"sample_{idx:06d}.wav"
                filepath = os.path.join(OUT_WAV_DIR, filename)
                sf.write(filepath, audio, SR, subtype="PCM_16")
                
                processed_samples.append({
                    'path': filepath,
                    'text': sample_info['text']
                })
                
            # اگر audio یک dict است
            elif isinstance(audio_data, dict):
                print(f"Audio is dict with keys: {audio_data.keys()}")
                
            else:
                print(f"Unknown audio type: {type(audio_data)}")
                
        except Exception as e:
            print(f"Error processing audio {idx}: {e}")
            continue
    
    print(f"\n✅ Successfully processed {len(processed_samples)} samples")
    
    if len(processed_samples) > 0:
        # ذخیره فایل‌لیست تست
        with open(os.path.join(OUT_FL_DIR, "test_samples.txt"), "w", encoding="utf-8") as f:
            for sample in processed_samples:
                f.write(f"{sample['path']}|{sample['text']}\n")
        print(f"📝 Test filelist saved to {OUT_FL_DIR}/test_samples.txt")

if __name__ == "__main__":
    main()
