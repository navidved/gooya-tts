import json
import torch
import os

def calculate_optimal_batch_size(vram_gb=140):
    """محاسبه batch_size بهینه بر اساس VRAM"""
    # تخمین: هر sample حدود 1-1.5GB در fp16/bf16
    # با 140GB می‌توانیم حداقل 96 sample
    
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
            "bf16_run": True if vram_gb >= 80 else False,  # BF16 برای H100/A100
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 10,
            "c_mel": 45,
            "c_kl": 1.0,
            "grad_clip_thresh": 5.0,
            "num_workers": min(32, os.cpu_count() or 8),
            "checkpoint_interval": 5000,
            "use_flash_attn": vram_gb >= 80,  # Flash Attention برای H100/A100
        },
        "data": {
            "training_files": "filelists/train.txt",
            "validation_files": "filelists/val.txt",
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
            "inter_channels": 256 if vram_gb >= 80 else 192,
            "hidden_channels": 256 if vram_gb >= 80 else 192,
            "filter_channels": 1024 if vram_gb >= 80 else 768,
            "n_heads": 4 if vram_gb >= 80 else 2,
            "n_layers": 8 if vram_gb >= 80 else 6,
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

    
    # نسخه استاندارد
    with open("~/vits2_pytorch/configs/vits2_persian.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # نسخه با فونم (اگر نیاز است)
    config_phoneme = config.copy()
    config_phoneme["data"]["training_files"] = "filelists/train_phoneme.txt"
    config_phoneme["data"]["validation_files"] = "filelists/val_phoneme.txt"
    config_phoneme["data"]["text_cleaners"] = []
    config_phoneme["data"]["cleaned_text"] = True
    
    with open("~/vits2_pytorch/configs/vits2_persian_phoneme.json", "w") as f:
        json.dump(config_phoneme, f, indent=2)
    
    print(f"\n✅ Configs created:")
    print(f"  - ~/vits2_pytorch/configs/vits2_persian.json (batch_size={config['train']['batch_size']})")
    print(f"  - ~/vits2_pytorch/configs/vits2_persian_phoneme.json")
    
    # نمایش توصیه‌ها
    print(f"\n📊 Recommendations for your system:")
    print(f"  - Optimal batch size: {optimal_batch}")
    print(f"  - BF16 training: {'Yes ✅' if config['train']['bf16_run'] else 'No'}")
    print(f"  - Flash Attention: {'Yes ✅' if vram_gb >= 80 else 'No'}")
    print(f"  - Model size: {'Large' if vram_gb >= 80 else 'Standard'}")
    
    # تخمین سرعت آموزش
    samples_per_sec = optimal_batch * 2  # تخمینی
    total_samples = 6500  # از دیتاست شما
    steps_per_epoch = total_samples / optimal_batch
    time_per_epoch = steps_per_epoch / samples_per_sec / 60  # دقیقه
    
    print(f"\n⏱️ Training estimates:")
    print(f"  - Steps per epoch: {steps_per_epoch:.0f}")
    print(f"  - Time per epoch: ~{time_per_epoch:.1f} minutes")
    print(f"  - Total training time (200k steps): ~{200000/samples_per_sec/3600:.1f} hours")

if __name__ == "__main__":
    create_optimized_config()