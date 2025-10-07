#!/bin/bash
# Train a small GPT model on Shakespeare with proper logging

python3 train_gpt2.py \
  --dataset=shakespeare \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --batch_size=12 \
  --block_size=256 \
  --max_iters=5000 \
  --eval_interval=500 \
  --log_interval=10 \
  --always_save_checkpoint=True \
  2>&1 | tee out/training_$(date +%Y%m%d_%H%M%S).log
