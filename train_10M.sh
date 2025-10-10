#!/bin/bash
# Train a ~10M parameter GPT model
# Estimated time: ~8 hours on M2 Ultra

echo "🚀 Training 10M Parameter GPT Model"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Layers: 12"
echo "  Heads: 6"
echo "  Embedding: 384"
echo "  Total params: ~12.5M"
echo "  Non-embedding: ~2.7M"
echo ""
echo "Estimated time: 8 hours"
echo ""

python3 train_gpt2.py \
  --dataset=shakespeare \
  --n_layer=12 \
  --n_head=6 \
  --n_embd=384 \
  --batch_size=12 \
  --block_size=1024 \
  --max_iters=20000 \
  --eval_interval=1000 \
  --log_interval=10 \
  --always_save_checkpoint=True \
  2>&1 | tee out/training_10M_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Training complete!"
echo "📊 Check out/training_10M_*.log for full logs"
echo "🎯 Model saved to out/ckpt.pt"
