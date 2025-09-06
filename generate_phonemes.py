import os
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from tqdm import tqdm

def create_phoneme_filelists():
    """تبدیل متن به فونم"""
    
    # تنظیمات phonemizer
    backend = EspeakBackend('fa', preserve_punctuation=True)
    separator = Separator(phone=' ', word='| ', syllable='')
    
    for split in ['train', 'val']:
        input_file = f"filelists/{split}.txt"
        output_file = f"filelists/{split}_phoneme.txt"
        
        print(f"Processing {split} split...")
        
        with open(input_file, 'r', encoding='utf-8') as inf:
            lines = inf.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            for line in tqdm(lines, desc=f"Phonemizing {split}"):
                path, text = line.strip().split('|')
                
                try:
                    phones = phonemize(
                        text,
                        language='fa',
                        backend='espeak',
                        separator=separator,
                        strip=True,
                        preserve_punctuation=True
                    )
                    phones = ' '.join(phones.split())  # نرمال‌سازی فاصله‌ها
                    outf.write(f"{path}|{phones}\n")
                except Exception as e:
                    print(f"Error phonemizing: {text[:50]}... - {e}")
                    outf.write(f"{path}|{text}\n")  # fallback to original
        
        print(f"✅ Created {output_file}")

if __name__ == "__main__":
    create_phoneme_filelists()