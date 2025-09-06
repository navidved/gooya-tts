# optimize_config.py - نسخه اصلاح شده
import json
import torch
import os
from pathlib import Path

def calculate_optimal_batch_size(vram_gb=140):
    """محاسبه batch_size بهینه بر اساس VRAM"""
    if vram_gb >= 140:
        return 128
    elif vram_gb >= 80:
        return 64
    elif vram_gb >= 40:
        return 32
    else:
        return 16

def create_optimized_config():
    # بررسی GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 VRAM: {vram_gb:.1f} GB")
    else:
        print("⚠️ No GPU detected, using default config")
        vram_gb = 0
    
    # محاسبه تنظیمات بهینه
    optimal_batch = calculate_optimal_batch_size(vram_gb)
    
    # مسیرهای کامل برای فایل‌های داده
    gooya_path = Path.home() / "gooya-tts"
    train_path = str(gooya_path / "filelists/train.txt")
    val_path = str(gooya_path / "filelists/val.txt")
    train_phoneme_path = str(gooya_path / "filelists/train_phoneme.txt")
    val_phoneme_path = str(gooya_path / "filelists/val_phoneme.txt")
    
    config = {
        "train": {
            "log_interval": 100,
            "eval_interval": 1000,
            "save_interval": 5000,
            "seed": 1234,
            "epochs": 2000,
            "learning_rate": 0.0002,
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": optimal_batch,
            "fp16_run": False,
            "bf16_run": True,  # BF16 برای H200
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 10,
            "c_mel": 45,
            "c_kl": 1.0,
            "grad_clip_thresh": 5.0,
            "num_workers": min(32, os.cpu_count() or 8),
            "checkpoint_interval": 5000,
        },
        "data": {
            "training_files": train_path,  # مسیر کامل
            "validation_files": val_path,  # مسیر کامل
            "text_cleaners": ["basic_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 11025.0,
            "add_blank": True,
            "n_speakers": 0,
            "cleaned_text": False
        },
        "model": {
            "use_mel_posterior_encoder": True,
            "use_transformer_flows": True,
            "transformer_flow_type": "fft",
            "use_spk_conditioned_encoder": False,
            "use_noise_scaled_mas": True,
            "use_duration_discriminator": True,
            "inter_channels": 256,
            "hidden_channels": 256,
            "filter_channels": 1024,
            "n_heads": 4,
            "n_layers": 8,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "use_sdp": True
        }
    }
    
    # مسیر صحیح برای ذخیره در vits2_pytorch
    vits2_path = Path.home() / "vits2_pytorch" / "configs"
    vits2_path.mkdir(parents=True, exist_ok=True)
    
    # ذخیره در vits2_pytorch/configs
    config_path = vits2_path / "vits2_persian.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # نسخه با فونم
    config_phoneme = config.copy()
    config_phoneme["data"]["training_files"] = train_phoneme_path  # مسیر کامل
    config_phoneme["data"]["validation_files"] = val_phoneme_path  # مسیر کامل
    config_phoneme["data"]["text_cleaners"] = []
    config_phoneme["data"]["cleaned_text"] = True
    
    config_phoneme_path = vits2_path / "vits2_persian_phoneme.json"
    with open(config_phoneme_path, "w") as f:
        json.dump(config_phoneme, f, indent=2)
    
    print(f"\n✅ Configs created successfully:")
    print(f"  1. {config_path}")
    print(f"  2. {config_phoneme_path}")
    
    # نمایش توصیه‌ها برای H200
    print(f"\n📊 Optimized settings for NVIDIA H200 (140GB):")
    print(f"  - Batch size: {optimal_batch}")
    print(f"  - BF16 training: ✅ Enabled")
    print(f"  - Model channels: 256 (Large)")
    print(f"  - Attention heads: 4")
    print(f"  - Transformer layers: 8")
    print(f"  - Workers: {config['train']['num_workers']}")
    
    # تخمین سرعت آموزش
    print(f"\n⏱️ Training estimates for H200:")
    print(f"  - Steps per epoch: ~{6500/optimal_batch:.0f}")
    print(f"  - Estimated speed: ~{optimal_batch*2:.0f} samples/sec")
    print(f"  - Time per epoch: ~{6500/optimal_batch/60:.1f} minutes")
    print(f"  - To 200k steps: ~15-20 hours")
    
    print(f"\n💡 Tips for H200:")
    print(f"  1. You can try batch_size=160 or even 192")
    print(f"  2. Use torch.compile() for 20-30% speedup")
    print(f"  3. Enable Flash Attention 2 in training script")

if __name__ == "__main__":
    create_optimized_config()