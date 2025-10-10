#!/bin/bash
# Complete pipeline for training a lyrics generation model
# Optimized for 10M parameters, ~8 hour training

echo "🎵 Song Lyrics GPT Training Pipeline"
echo "===================================="
echo ""

# Step 1: Create sample dataset
echo "📝 Step 1: Creating sample lyrics dataset..."
python3 create_sample_lyrics.py
echo ""

# Step 2: Prepare dataset
echo "🔄 Step 2: Preparing dataset for training..."
python3 prepare_custom_data.py \
  --input sample_lyrics.txt \
  --dataset_name lyrics
echo ""

# Step 3: Train model
echo "🚀 Step 3: Training lyrics generation model..."
echo ""
echo "Configuration:"
echo "  Model size: ~10M parameters"
echo "  Layers: 12"
echo "  Heads: 6"
echo "  Embedding: 384"
echo "  Training time: ~4-6 hours"
echo ""
read -p "Continue with training? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Training cancelled. You can run manually with:"
    echo "  python train_gpt2.py --dataset=lyrics --n_layer=12 --n_head=6 --n_embd=384 --max_iters=15000"
    exit 0
fi

python3 train_gpt2.py \
  --dataset=lyrics \
  --n_layer=12 \
  --n_head=6 \
  --n_embd=384 \
  --batch_size=12 \
  --block_size=512 \
  --max_iters=15000 \
  --eval_interval=1000 \
  --log_interval=10 \
  --dropout=0.1 \
  --always_save_checkpoint=True \
  2>&1 | tee out/training_lyrics_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Training complete!"
echo ""
echo "🎤 Now generate lyrics with:"
echo "   python generate_interactive.py"
echo ""
echo "💡 Try these prompts:"
echo '   "[Verse 1]"'
echo '   "[Chorus]"'
echo '   "Love is"'
echo '   "Tonight we"'
echo ""
echo "🎨 Adjust creativity with temperature:"
echo "   0.7 = Conservative"
echo "   0.9 = Balanced (recommended)"
echo "   1.1 = Very creative"
