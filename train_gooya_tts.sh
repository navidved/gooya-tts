#!/bin/bash

# رنگ‌ها
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Gooya TTS Training${NC}"

# مسیرها
VITS2_DIR="../vits2_pytorch"
CURRENT_DIR=$(pwd)

# فعال‌سازی venv - بررسی هر دو مکان
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Activated venv from gooya-tts${NC}"
elif [ -d "$VITS2_DIR/venv" ]; then
    source $VITS2_DIR/venv/bin/activate
    echo -e "${GREEN}✓ Activated venv from vits2_pytorch${NC}"
else
    echo -e "${YELLOW}⚠ No venv found, using system Python${NC}"
fi

# تنظیمات محیطی H200
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDA_ARCH_LIST="9.0"

# تنظیمات مدل
MODEL_NAME="gooya_tts_$(date +%Y%m%d_%H%M%S)"
CONFIG="configs/vits2_persian.json"

# بررسی وجود config
if [ ! -f "$VITS2_DIR/$CONFIG" ]; then
    echo -e "${RED}❌ Config not found at $VITS2_DIR/$CONFIG${NC}"
    echo -e "${YELLOW}Please run: python optimize_config.py${NC}"
    exit 1
fi

# نمایش اطلاعات
echo -e "${GREEN}Model: $MODEL_NAME${NC}"
echo -e "${GREEN}Config: $CONFIG${NC}"

# ایجاد دایرکتوری لاگ
mkdir -p $VITS2_DIR/logs/$MODEL_NAME

# شروع TensorBoard
pkill -f "tensorboard" 2>/dev/null
cd $VITS2_DIR && tensorboard --logdir=logs --port=6006 --bind_all &
TB_PID=$!
echo -e "${YELLOW}TensorBoard: http://localhost:6006 (PID: $TB_PID)${NC}"

# رفتن به vits2_pytorch و شروع training
cd $VITS2_DIR

echo -e "${GREEN}Starting training...${NC}"
python train.py -c $CONFIG -m $MODEL_NAME 2>&1 | tee logs/$MODEL_NAME/training.log

# برگشت به gooya-tts
cd $CURRENT_DIR

echo -e "${GREEN}✅ Training finished!${NC}"
echo -e "${YELLOW}Checkpoints saved in: $VITS2_DIR/logs/$MODEL_NAME/${NC}"