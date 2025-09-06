# train_with_monitoring.sh
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# شروع tensorboard
tensorboard --logdir=logs --port=6006 &

# آموزش با resume capability
python train.py \
    -c configs/vits2_fa_ali.json \
    -m ali_tts_vits2 \
    --checkpoint_path "" \
    2>&1 | tee training.log