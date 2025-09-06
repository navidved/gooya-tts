# check_setup.py - نسخه اصلاح شده
import os
import json
from pathlib import Path

def check_setup():
    print("🔍 Checking setup...")
    
    # بررسی config در vits2_pytorch
    config_path = Path.home() / "vits2_pytorch/configs/vits2_persian.json"
    
    if config_path.exists():
        print(f"✅ Config found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # بررسی فایل‌های داده
        train_file = config['data']['training_files']
        val_file = config['data']['validation_files']
        
        if os.path.exists(train_file):
            train_lines = len(open(train_file).readlines())
            print(f"✅ Train file: {train_file} ({train_lines} samples)")
        else:
            print(f"❌ Train file not found: {train_file}")
        
        if os.path.exists(val_file):
            val_lines = len(open(val_file).readlines())
            print(f"✅ Val file: {val_file} ({val_lines} samples)")
        else:
            print(f"❌ Val file not found: {val_file}")
        
        # بررسی یک نمونه WAV
        if os.path.exists(train_file):
            first_line = open(train_file).readline().strip()
            wav_path = first_line.split('|')[0]
            if os.path.exists(wav_path):
                print(f"✅ Sample WAV exists: {wav_path}")
            else:
                print(f"❌ Sample WAV not found: {wav_path}")
    else:
        print(f"❌ Config not found: {config_path}")
    
    # بررسی فایل‌های محلی در gooya-tts
    local_files = {
        "Train data (local)": Path("filelists/train.txt"),
        "Val data (local)": Path("filelists/val.txt"),
        "WAV folder": Path("data_fa/wavs")
    }
    
    for name, path in local_files.items():
        if path.exists():
            if path.is_file():
                print(f"✅ {name}: {path}")
            else:
                count = len(list(path.glob("*.wav")))
                print(f"✅ {name}: {count} files")
        else:
            print(f"❌ {name}: NOT FOUND")
    
    # بررسی train.py
    train_script = Path.home() / "vits2_pytorch/train.py"
    if train_script.exists():
        print(f"✅ Train script: {train_script}")
    else:
        print(f"❌ Train script: NOT FOUND")
    
    # بررسی GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("❌ No GPU available")
    except:
        print("⚠️ PyTorch not properly installed")
    
    print("\n🎉 Setup check complete!")

if __name__ == "__main__":
    check_setup()