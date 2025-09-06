# prepare_data_fa.py - نسخه بهبود یافته
import os, re, unicodedata, random, soundfile as sf, librosa
from datasets import load_dataset, Audio
import numpy as np
from tqdm import tqdm  # برای نمایش پیشرفت

HF_DATASET = "navidved/approved-tts-dataset"
OUT_WAV_DIR = "data_fa/wavs"
OUT_FL_DIR  = "filelists"
SR = 22050
MIN_DUR = 0.6
MAX_DUR = 15.5

os.makedirs(OUT_WAV_DIR, exist_ok=True)
os.makedirs(OUT_FL_DIR, exist_ok=True)

# نگاشت‌های کاراکتری
ARABIC_TO_FA = {'\u064A': 'ی', '\u0643': 'ک', '\u06C0': 'ه', '\u06CC': 'ی'}
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
LATIN_DIGITS   = "0123456789"
DIGIT_MAP = {ord(p): LATIN_DIGITS[i] for i,p in enumerate(PERSIAN_DIGITS)}

DIACRITICS = ''.join(chr(c) for c in range(0x064B, 0x065F+1)) + "\u0670"
DIAC_RE = re.compile(f"[{re.escape(DIACRITICS)}]")

def fa_normalize(s):
    # حذف کاراکترهای کنترلی
    s = ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C')
    
    s = ''.join(ARABIC_TO_FA.get(ch, ch) for ch in s)
    s = DIAC_RE.sub("", s)
    s = s.translate(DIGIT_MAP)
    s = s.replace("\u0640", "")  # تطویل
    s = s.replace("\u200c", " ")  # نیم‌فاصله به فاصله
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[«»]", "\"", s)
    
    # حذف علائم اضافی که ممکنه مشکل ایجاد کنند
    s = re.sub(r'[^\u0600-\u06FF\u0020-\u007E\s]', '', s)
    
    return s

def main():
    # لود دیتاست
    print("Loading dataset...")
    ds = load_dataset(HF_DATASET, split="train", use_auth_token=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))

    rows = []
    skipped_short = 0
    skipped_long = 0
    
    print(f"Processing {len(ds)} samples...")
    for idx, item in enumerate(tqdm(ds)):
        txt = fa_normalize(item["sentence"])
        
        # بررسی طول متن
        if len(txt) < 3 or len(txt) > 300:
            continue
            
        audio = item["audio"]["array"]
        
        # حذف سکوت ابتدا و انتها
        audio = librosa.effects.trim(audio, top_db=20)[0]
        
        # نرمالیزه صدا
        peak = np.abs(audio).max()
        if peak > 0:
            audio = (0.95 / peak) * audio
        
        # ایجاد نام یکتا برای فایل
        base = f"sample_{idx:06d}"
        out_wav = os.path.join(OUT_WAV_DIR, base + ".wav")
        
        dur = len(audio) / SR
        
        if dur < MIN_DUR:
            skipped_short += 1
            continue
        elif dur > MAX_DUR:
            skipped_long += 1
            continue
            
        sf.write(out_wav, audio, SR, subtype="PCM_16")
        rows.append((out_wav, txt))

    print(f"Skipped: {skipped_short} too short, {skipped_long} too long")
    
    # split (95/5 برای دیتاست کوچک بهتره)
    random.seed(42)
    random.shuffle(rows)
    n = len(rows)
    n_val = max(200, int(0.05*n))  # حداقل 200 نمونه validation
    val = rows[:n_val]
    train = rows[n_val:]

    with open(os.path.join(OUT_FL_DIR, "ali_train.txt"), "w", encoding="utf-8") as f:
        for p,t in train:
            f.write(f"{p}|{t}\n")
    with open(os.path.join(OUT_FL_DIR, "ali_val.txt"), "w", encoding="utf-8") as f:
        for p,t in val:
            f.write(f"{p}|{t}\n")

    print(f"✅ Prepared {len(train)} train and {len(val)} val samples at {SR} Hz.")

if __name__ == "__main__":
    main()