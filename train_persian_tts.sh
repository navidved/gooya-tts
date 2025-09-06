#!/bin/bash

# رنگ‌ها برای نمایش بهتر
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Persian TTS Training on H200${NC}"

# بررسی دایرکتوری
if [ ! -f "train.py" ]; then
    echo -e "${RED}Error: train.py not found! Are you in vits2_pytorch directory?${NC}"
    echo "Current directory: $(pwd)"
    echo "Please run: cd ~/vits2_pytorch"
    exit 1
fi

# تنظیمات محیطی برای H200
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# بهینه‌سازی‌های H200
export TORCH_CUDA_ARCH_LIST="9.0"  # برای H200
export CUDA_CACHE_DISABLE=0
export TORCH_BACKENDS_CUDNN_BENCHMARK=1
export OMP_NUM_THREADS=32

# نام مدل و config
MODEL_NAME="persian_tts_h200_$(date +%Y%m%d_%H%M%S)"

# انتخاب config بر اساس وجود فایل‌های فونم
if [ -f "/home/modir/gooya-tts/filelists/train_phoneme.txt" ]; then
    CONFIG="configs/vits2_persian_phoneme.json"
    echo -e "${GREEN}Using phoneme config${NC}"
else
    CONFIG="configs/vits2_persian.json"
    echo -e "${YELLOW}Using text config (no phonemes)${NC}"
fi

# بررسی وجود config
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

# بررسی فایل‌های دیتاست
echo -e "${YELLOW}Checking dataset files...${NC}"
if [ -f "/home/modir/gooya-tts/filelists/train.txt" ]; then
    TRAIN_LINES=$(wc -l < /home/modir/gooya-tts/filelists/train.txt)
    VAL_LINES=$(wc -l < /home/modir/gooya-tts/filelists/val.txt)
    echo -e "${GREEN}✓ Train samples: $TRAIN_LINES${NC}"
    echo -e "${GREEN}✓ Val samples: $VAL_LINES${NC}"
else
    echo -e "${RED}Error: Dataset files not found!${NC}"
    exit 1
fi

# ایجاد دایرکتوری‌ها
mkdir -p logs/$MODEL_NAME
mkdir -p checkpoints/$MODEL_NAME

# ذخیره اطلاعات سیستم
echo "=== System Info ===" > logs/$MODEL_NAME/system_info.txt
nvidia-smi >> logs/$MODEL_NAME/system_info.txt
echo "" >> logs/$MODEL_NAME/system_info.txt
echo "Python: $(python --version)" >> logs/$MODEL_NAME/system_info.txt
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')" >> logs/$MODEL_NAME/system_info.txt
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')" >> logs/$MODEL_NAME/system_info.txt
echo "Config: $CONFIG" >> logs/$MODEL_NAME/system_info.txt
echo "Model: $MODEL_NAME" >> logs/$MODEL_NAME/system_info.txt

# کپی config برای reference
cp $CONFIG logs/$MODEL_NAME/config.json

# Kill existing TensorBoard if running
pkill -f "tensorboard --logdir=logs" 2>/dev/null

# شروع TensorBoard
echo -e "${YELLOW}Starting TensorBoard...${NC}"
tensorboard --logdir=logs --port=6006 --bind_all --reload_interval=30 &
TB_PID=$!
echo "TensorBoard PID: $TB_PID (http://localhost:6006)"

# ایجاد اسکریپت resume برای ادامه آموزش در صورت قطع
cat > resume_training.sh << 'RESUME'
#!/bin/bash
LATEST_CKPT=$(ls -t logs/MODEL_NAME_PLACEHOLDER/G_*.pth 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from: $LATEST_CKPT"
    python train.py \
        -c CONFIG_PLACEHOLDER \
        -m MODEL_NAME_PLACEHOLDER \
        --checkpoint_path "$LATEST_CKPT"
else
    echo "No checkpoint found to resume from"
fi
RESUME

sed -i "s|MODEL_NAME_PLACEHOLDER|$MODEL_NAME|g" resume_training.sh
sed -i "s|CONFIG_PLACEHOLDER|$CONFIG|g" resume_training.sh
chmod +x resume_training.sh

# نمایش اطلاعات
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Model: $MODEL_NAME"
echo -e "  Config: $CONFIG"
echo -e "  GPU: NVIDIA H200 (140GB)"
echo -e "  Logs: logs/$MODEL_NAME/"
echo -e "${GREEN}========================================${NC}"

# شروع آموزش با error handling
echo -e "${GREEN}Starting training...${NC}"

python train.py \
    -c $CONFIG \
    -m $MODEL_NAME \
    2>&1 | tee logs/$MODEL_NAME/training.log

# بررسی exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
else
    echo -e "${RED}✗ Training failed or interrupted${NC}"
    echo -e "${YELLOW}To resume training, run: ./resume_training.sh${NC}"
fi

# نمایش آمار نهایی
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Summary:${NC}"
if [ -f "logs/$MODEL_NAME/training.log" ]; then
    echo "Last 10 lines of log:"
    tail -10 logs/$MODEL_NAME/training.log
fi

# لیست checkpoints
echo -e "\n${GREEN}Saved checkpoints:${NC}"
ls -lh logs/$MODEL_NAME/*.pth 2>/dev/null | tail -5

echo -e "\n${YELLOW}TensorBoard is still running on http://localhost:6006${NC}"
echo -e "${YELLOW}To stop it: kill $TB_PID${NC}"