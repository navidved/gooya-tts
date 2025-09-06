import os
from pathlib import Path

checks = {
    "Config": Path.home() / "vits2_pytorch/configs/vits2_persian.json",
    "Train data": Path("filelists/train.txt"),
    "Val data": Path("filelists/val.txt"),
    "WAV folder": Path("data_fa/wavs"),
    "Train.py": Path.home() / "vits2_pytorch/train.py"
}

print("🔍 Final checks:")
for name, path in checks.items():
    if path.exists():
        if path.is_file():
            print(f"✅ {name}: {path}")
        else:
            count = len(list(path.glob("*.wav")))
            print(f"✅ {name}: {count} files")
    else:
        print(f"❌ {name}: NOT FOUND")

import torch
print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("\n🎉 Ready to train!")
