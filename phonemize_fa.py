# phonemize_fa.py - فونم‌سازی بهتر برای فارسی
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

def phonemize_persian(text):
    backend = EspeakBackend('fa', preserve_punctuation=True)
    separator = Separator(phone=' ', word='| ', syllable='')
    
    phones = phonemize(
        text,
        language='fa',
        backend='espeak',
        separator=separator,
        strip=True,
        preserve_punctuation=True
    )
    
    # پاکسازی اضافی برای فونم‌های فارسی
    phones = re.sub(r'\s+', ' ', phones)
    return phones

# اعمال روی فایل‌لیست‌ها
def create_phoneme_filelists():
    for split in ['train', 'val']:
        input_file = f"filelists/ali_{split}.txt"
        output_file = f"filelists/ali_{split}_phoneme.txt"
        
        with open(input_file, 'r', encoding='utf-8') as inf:
            with open(output_file, 'w', encoding='utf-8') as outf:
                for line in inf:
                    path, text = line.strip().split('|')
                    phones = phonemize_persian(text)
                    outf.write(f"{path}|{phones}\n")
        
        print(f"Created {output_file}")

create_phoneme_filelists()