# 📚 راهنمای کامل آموزش TTS فارسی با VITS2 روی HuggingFace

## 📋 مشخصات پروژه
- دیتاست: 6795 نمونه × 15 ثانیه = ~28 ساعت صدا
- سرور: H200 140GB VRAM + 197GB RAM + 1TB Disk
- مدل: VITS2 Single-Speaker Persian TTS

---

## 🚀 مرحله 1: راه‌اندازی اولیه VM

```bash
# بروزرسانی سیستم و نصب ابزارهای پایه

sudo apt-get update && sudo apt-get upgrade -y

sudo apt install -y \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libncurses5-dev \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libffi-dev \
  liblzma-dev \
  uuid-dev \
  wget \
  curl \
  git

sudo apt-get install -y \
    build-essential \
    cmake \
    nvtop \
    tmux \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    libespeak-ng1 \
    python3-pip


# نصب نسخه مناسب python
نصب python 3.10 با pyenv

# بررسی GPU
nvidia-smi
```

---

## 🐍 مرحله 2: نصب VENV و ایجاد محیط

```bash
# کلون repository
git clone https://github.com/p0p4k/vits2_pytorch.git

# ایجاد محیط مجازی
cd ~/vits2_pytorch
python3 -m venv venv

# فعال‌سازی
source venv/bin/activate

# آپگرید pip
pip install --upgrade pip setuptools wheel

# برای CUDA 12.8 و H200
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# بررسی نصب صحیح
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 📦 مرحله 3: نصب VITS2 و وابستگی‌ها

```bash
# نصب requirements
pip install -r requirements.txt

# نصب پکیج‌های اضافی
pip install \
    huggingface_hub \
    datasets \
    soundfile \
    librosa \
    tensorboard \
    wandb \
    matplotlib \
    phonemizer \
    hazm \
    tqdm \
    jiwer \
    torchmetrics

# Build monotonic alignment
cd monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd ..

# اجرا tmux
tmux

# ورود به HuggingFace
huggingface-cli login
# توکن خود را وارد کنید
```

---

## 🎵 مرحله 4: آماده‌سازی دیتاست

### ایجاد اسکریپت آماده‌سازی داده

```bash
python prepare_dataset.py
```

### 4.2 - تولید فونم (اختیاری اما توصیه می‌شود)

```bash
cat > generate_phonemes.py << 'EOF'
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
EOF

python generate_phonemes.py
```

---

## ⚙️ مرحله 5: پیکربندی مدل

```bash
cat > configs/vits2_persian.json << 'EOF'
{
  "train": {
    "log_interval": 100,
    "eval_interval": 1000,
    "seed": 1234,
    "epochs": 2000,
    "learning_rate": 0.0002,
    "betas": [0.8, 0.99],
    "eps": 1e-09,
    "batch_size": 32,
    "fp16_run": true,
    "lr_decay": 0.999875,
    "segment_size": 8192,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "grad_clip_thresh": 5.0,
    "num_workers": 8,
    "checkpoint_interval": 5000,
    "use_sr_rates": false
  },
  "data": {
    "training_files": "filelists/train_phoneme.txt",
    "validation_files": "filelists/val_phoneme.txt",
    "text_cleaners": [],
    "max_wav_value": 32768.0,
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 0,
    "cleaned_text": true
  },
  "model": {
    "use_mel_posterior_encoder": true,
    "use_transformer_flows": true,
    "transformer_flow_type": "fft",
    "use_spk_conditioned_encoder": false,
    "use_noise_scaled_mas": true,
    "use_duration_discriminator": true,
    "ms_istft_vits": false,
    "mb_istft_vits": false,
    "istft_vits": false,
    "subbands": 4,
    "gen_istft_n_fft": 16,
    "gen_istft_hop_size": 4,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "n_layers_q": 3,
    "use_spectral_norm": false,
    "use_sdp": true
  }
}
EOF
```

---

## 🎯 مرحله 6: شروع آموزش

### 6.1 - ایجاد اسکریپت آموزش

```bash
cat > train_model.sh << 'EOF'
#!/bin/bash

# تنظیمات محیطی
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# نام مدل
MODEL_NAME="persian_tts_vits2"
CONFIG="configs/vits2_persian.json"

# ایجاد دایرکتوری لاگ
mkdir -p logs/$MODEL_NAME

# شروع TensorBoard در background
echo "Starting TensorBoard..."
tensorboard --logdir=logs --port=6006 --bind_all &
TB_PID=$!
echo "TensorBoard PID: $TB_PID"

# شروع آموزش
echo "Starting training..."
python train.py \
    -c $CONFIG \
    -m $MODEL_NAME \
    2>&1 | tee logs/$MODEL_NAME/training.log

echo "Training completed!"
EOF

chmod +x train_model.sh
```

### 6.2 - اجرای آموزش در tmux

```bash
# ایجاد session جدید
tmux new -s vits2_training

# اجرای آموزش
./train_model.sh

# برای خروج از tmux: Ctrl+B then D
# برای برگشت: tmux attach -t vits2_training
```

---

## 📊 مرحله 7: مانیتورینگ

### 7.1 - اسکریپت مانیتورینگ real-time

```bash
cat > monitor_training.py << 'EOF'
import os
import time
import json
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def monitor_training(log_dir, model_name):
    """مانیتورینگ پیشرفت آموزش"""
    
    while True:
        try:
            # خواندن آخرین checkpoint
            ckpt_dir = f"logs/{model_name}"
            checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("G_") and f.endswith(".pth")]
            
            if checkpoints:
                latest_ckpt = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
                step = int(latest_ckpt.split("_")[1].split(".")[0])
                
                print(f"\n{'='*50}")
                print(f"Latest Checkpoint: {latest_ckpt}")
                print(f"Current Step: {step:,}")
                
                # محاسبه زمان تخمینی
                estimated_total_steps = 200000  # تخمین
                progress = (step / estimated_total_steps) * 100
                print(f"Progress: {progress:.2f}%")
                
                # خواندن loss از TensorBoard logs
                event_acc = EventAccumulator(log_dir)
                event_acc.Reload()
                
                if 'loss/g/total' in event_acc.Tags()['scalars']:
                    g_loss = event_acc.Scalars('loss/g/total')
                    if g_loss:
                        latest_g_loss = g_loss[-1].value
                        print(f"Generator Loss: {latest_g_loss:.4f}")
                
                if 'loss/d/total' in event_acc.Tags()['scalars']:
                    d_loss = event_acc.Scalars('loss/d/total')
                    if d_loss:
                        latest_d_loss = d_loss[-1].value
                        print(f"Discriminator Loss: {latest_d_loss:.4f}")
                
                print(f"{'='*50}")
                
        except Exception as e:
            print(f"Monitoring error: {e}")
        
        time.sleep(60)  # بروزرسانی هر 60 ثانیه

if __name__ == "__main__":
    monitor_training("logs/persian_tts_vits2", "persian_tts_vits2")
EOF

# اجرا در پس‌زمینه
python monitor_training.py &
```

### 7.2 - دسترسی به TensorBoard

```bash
# اگر روی سرور remote هستید:
# در لوکال:
ssh -L 6006:localhost:6006 user@server_ip

# سپس در مرورگر:
# http://localhost:6006
```

---

## 🧪 مرحله 8: تست مدل

### 8.1 - اسکریپت inference

```bash
cat > inference.py << 'EOF'
import torch
import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
import soundfile as sf
import os
import json

def load_model(checkpoint_path, config_path):
    """بارگذاری مدل"""
    
    with open(config_path, "r") as f:
        hps = json.load(f)
    hps = utils.HParams(hps)
    
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        hps.model
    ).cuda()
    
    _ = net_g.eval()
    _ = utils.load_checkpoint(checkpoint_path, net_g, None)
    
    return net_g, hps

def synthesize(text, model, hps, output_path="output.wav"):
    """تولید صدا از متن"""
    
    # تبدیل متن به sequence
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    
    text_norm = torch.LongTensor(text_norm)
    text_len = torch.LongTensor([text_norm.size(0)])
    
    with torch.no_grad():
        x = text_norm.cuda().unsqueeze(0)
        x_len = text_len.cuda()
        
        # تولید صدا
        audio = model.infer(
            x, 
            x_len, 
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0
        )[0][0, 0].data.cpu().float().numpy()
    
    # ذخیره
    sf.write(output_path, audio, hps.data.sampling_rate)
    print(f"✅ Audio saved to: {output_path}")
    
    return audio

def test_model():
    """تست مدل با نمونه‌های مختلف"""
    
    # مسیرها
    checkpoint = "logs/persian_tts_vits2/G_100000.pth"  # آخرین checkpoint
    config = "configs/vits2_persian.json"
    
    # بارگذاری مدل
    print("Loading model...")
    model, hps = load_model(checkpoint, config)
    
    # متن‌های تست
    test_texts = [
        "سلام، این یک تست برای سیستم تبدیل متن به گفتار فارسی است.",
        "هوش مصنوعی در حال تغییر دنیای ما است.",
        "امروز هوا بسیار زیبا و آفتابی است.",
        "کتاب خواندن یکی از بهترین سرگرمی‌ها است.",
        "تکنولوژی روز به روز پیشرفت می‌کند."
    ]
    
    # تولید صدا
    os.makedirs("test_outputs", exist_ok=True)
    
    for i, text in enumerate(test_texts):
        print(f"\nGenerating sample {i+1}...")
        print(f"Text: {text}")
        
        output_path = f"test_outputs/sample_{i+1}.wav"
        synthesize(text, model, hps, output_path)

if __name__ == "__main__":
    test_model()
EOF
```

### 8.2 - اجرای تست

```bash
# بعد از حداقل 50000 step
python inference.py
```

---

## 🎉 مرحله 9: Deployment

### 9.1 - ایجاد Web API

```bash
cat > app.py << 'EOF'
from flask import Flask, request, send_file, jsonify
import torch
import io
import base64
from inference import load_model, synthesize
import soundfile as sf
import numpy as np

app = Flask(__name__)

# بارگذاری مدل (یکبار)
print("Loading model...")
model, hps = load_model(
    "logs/persian_tts_vits2/G_best.pth",
    "configs/vits2_persian.json"
)
print("Model loaded!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/synthesize', methods=['POST'])
def synthesize_api():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # تولید صدا
        audio = synthesize(text, model, hps)
        
        # تبدیل به base64
        buffer = io.BytesIO()
        sf.write(buffer, audio, hps.data.sampling_rate, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            "audio": audio_base64,
            "sample_rate": hps.data.sampling_rate
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF

# نصب Flask
pip install flask

# اجرای سرور
python app.py
```

### 9.2 - ایجاد Gradio Interface

```bash
cat > gradio_app.py << 'EOF'
import gradio as gr
import torch
from inference import load_model, synthesize
import numpy as np

# بارگذاری مدل
print("Loading model...")
model, hps = load_model(
    "logs/persian_tts_vits2/G_best.pth",
    "configs/vits2_persian.json"
)

def tts_persian(text, speed=1.0, noise_scale=0.667):
    """تبدیل متن به گفتار"""
    
    if not text:
        return None
    
    # تولید صدا
    audio = synthesize(
        text, 
        model, 
        hps,
        length_scale=1/speed,
        noise_scale=noise_scale
    )
    
    return (hps.data.sampling_rate, audio)

# رابط کاربری
iface = gr.Interface(
    fn=tts_persian,
    inputs=[
        gr.Textbox(
            label="متن فارسی",
            placeholder="متن خود را اینجا وارد کنید...",
            lines=3
        ),
        gr.Slider(
            minimum=0.5,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="سرعت گفتار"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.667,
            step=0.01,
            label="تنوع صدا"
        )
    ],
    outputs=gr.Audio(label="صدای تولید شده"),
    title="🎙️ Persian Text-to-Speech (VITS2)",
    description="سیستم تبدیل متن به گفتار فارسی با استفاده از VITS2",
    examples=[
        ["سلام، به سیستم تبدیل متن به گفتار فارسی خوش آمدید.", 