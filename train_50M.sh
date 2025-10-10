#!/bin/bash
# Train a ~50M parameter GPT model
# Estimated time: ~2 days on M2 Ultra

echo "🚀 Training 50M Parameter GPT Model"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Layers: 8"
echo "  Heads: 8"
echo "  Embedding: 512"
echo "  Total params: ~51M"
echo "  Non-embedding: ~25M"
echo ""
echo "⚠️  Estimated time: 2 days"
echo "⚠️  This will use significant GPU resources"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

python3 train_gpt2.py \
  --dataset=shakespeare \
  --n_layer=8 \
  --n_head=8 \
  --n_embd=512 \
  --batch_size=12 \
  --block_size=1024 \
  --max_iters=50000 \
  --eval_interval=2000 \
  --log_interval=50 \
  --always_save_checkpoint=True \
  --dropout=0.1 \
  2>&1 | tee out/training_50M_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Training complete!"
echo "📊 Check out/training_50M_*.log for full logs"
echo "🎯 Model saved to out/ckpt.pt"
