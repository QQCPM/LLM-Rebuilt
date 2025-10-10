# 🚀 GPT Training Guide - Different Sizes & Use Cases

## 📏 Model Size Guide

### How to Get ~10M Parameters

**Option 1: Balanced (Recommended for 10M)**
```bash
python train_gpt2.py \
  --n_layer=12 \
  --n_head=6 \
  --n_embd=384 \
  --max_iters=20000 \
  --eval_interval=1000
```
→ **12.5M total, 2.7M non-embedding** parameters

**Option 2: Deeper (Better for complex patterns)**
```bash
python train_gpt2.py \
  --n_layer=16 \
  --n_head=8 \
  --n_embd=256 \
  --max_iters=20000 \
  --eval_interval=1000
```
→ **25.7M total, 12.6M non-embedding** parameters

**Option 3: Wider (Better for vocabulary)**
```bash
python train_gpt2.py \
  --n_layer=8 \
  --n_head=8 \
  --n_embd=512 \
  --max_iters=20000 \
  --eval_interval=1000
```
→ **51.5M total, 25.2M non-embedding** parameters

---

## 🎯 Different Training Tasks

### 1. **Code Generation (Python, JavaScript, etc.)**

**Prepare your code dataset:**
```bash
# Collect all Python files in your project
cat your_project/**/*.py > code_dataset.txt

# Or for JavaScript
cat your_project/**/*.js > code_dataset.txt

# Prepare for training
python prepare_custom_data.py \
  --input code_dataset.txt \
  --dataset_name my_code
```

**Train:**
```bash
python train_gpt2.py \
  --dataset=my_code \
  --n_layer=12 \
  --n_head=8 \
  --n_embd=512 \
  --max_iters=30000 \
  --block_size=512
```

**What you can do:**
- Auto-complete code
- Generate function implementations
- Fix bugs (with enough training)

---

### 2. **Conversation / Chat Bot**

**Prepare conversation data:**
```python
# Create a file with this format:
"""
User: Hello, how are you?
Bot: I'm doing well, thank you! How can I help you today?

User: Tell me a joke.
Bot: Why did the programmer quit his job? Because he didn't get arrays!

User: What's the weather like?
Bot: I don't have access to weather data, but I can help with other questions!
"""
```

**Prepare:**
```bash
python prepare_custom_data.py \
  --input conversations.txt \
  --dataset_name chatbot
```

**Train:**
```bash
python train_gpt2.py \
  --dataset=chatbot \
  --n_layer=8 \
  --n_head=8 \
  --n_embd=256 \
  --max_iters=15000
```

**Generate conversations:**
```python
# In generate_interactive.py, use prompts like:
# "User: Hello\nBot:"
```

---

### 3. **Creative Writing / Story Generation**

**Prepare stories:**
```bash
# Download books from Project Gutenberg or use your own
cat story1.txt story2.txt story3.txt > stories.txt

python prepare_custom_data.py \
  --input stories.txt \
  --dataset_name stories
```

**Train:**
```bash
python train_gpt2.py \
  --dataset=stories \
  --n_layer=12 \
  --n_head=12 \
  --n_embd=768 \
  --max_iters=50000 \
  --block_size=1024
```

---

### 4. **Domain-Specific Text (Medical, Legal, Technical)**

**Example: Medical notes**
```bash
# Collect domain-specific text
cat medical_notes/*.txt > medical_corpus.txt

python prepare_custom_data.py \
  --input medical_corpus.txt \
  --dataset_name medical
```

**Train with dropout (important for domain-specific!):**
```bash
python train_gpt2.py \
  --dataset=medical \
  --n_layer=12 \
  --n_head=8 \
  --n_embd=512 \
  --dropout=0.2 \
  --max_iters=25000
```

---

### 5. **Math / Arithmetic**

**Create math problems dataset:**
```python
# Generate simple math problems
import random

with open('math_problems.txt', 'w') as f:
    for _ in range(10000):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        f.write(f"Q: What is {a} + {b}?\nA: {a + b}\n\n")
        f.write(f"Q: What is {a} - {b}?\nA: {a - b}\n\n")
        f.write(f"Q: What is {a} * {b}?\nA: {a * b}\n\n")
```

**Prepare & Train:**
```bash
python prepare_custom_data.py \
  --input math_problems.txt \
  --dataset_name math

python train_gpt2.py \
  --dataset=math \
  --n_layer=8 \
  --n_head=4 \
  --n_embd=256 \
  --max_iters=20000
```

---

## 🎛️ Training Configuration Guide

### Fast Training (Testing)
```bash
--n_layer=4 --n_head=4 --n_embd=128 --max_iters=5000
# ~1M params, ~2 hours on M2 Ultra
```

### Medium Quality (10-25M params)
```bash
--n_layer=12 --n_head=6 --n_embd=384 --max_iters=20000
# ~12M params, ~8 hours on M2 Ultra
```

### High Quality (50-100M params)
```bash
--n_layer=12 --n_head=12 --n_embd=768 --max_iters=50000
# ~124M params (GPT-2 size), ~2-3 days on M2 Ultra
```

---

## 🔧 Important Hyperparameters

### Block Size (Context Length)
```bash
--block_size=256   # Short context (faster, less memory)
--block_size=512   # Medium (good for most tasks)
--block_size=1024  # Long context (slower, more memory)
```

### Dropout (Regularization)
```bash
--dropout=0.0   # No dropout (good for large datasets)
--dropout=0.1   # Light regularization (recommended)
--dropout=0.2   # Heavy regularization (small datasets)
```

### Batch Size (Speed vs Memory)
```bash
--batch_size=4   # Less memory, slower
--batch_size=12  # Balanced (default)
--batch_size=24  # More memory, faster (if you have 128GB RAM)
```

---

## 📊 Expected Training Times (M2 Ultra)

| Model Size | Iterations | Time | Final Loss (typical) |
|------------|-----------|------|---------------------|
| 1M params  | 5,000     | 2h   | 1.5-2.0 |
| 10M params | 20,000    | 8h   | 1.0-1.5 |
| 50M params | 50,000    | 2d   | 0.8-1.2 |
| 124M params| 100,000   | 4d   | 0.6-1.0 |

---

## 💡 Quick Start Examples

### Train 10M Model on Shakespeare
```bash
python train_gpt2.py \
  --dataset=shakespeare \
  --n_layer=12 --n_head=6 --n_embd=384 \
  --max_iters=20000 --eval_interval=1000 \
  2>&1 | tee out/training_10M.log
```

### Train Code Model
```bash
# First prepare your code
cat *.py > my_code.txt
python prepare_custom_data.py --input my_code.txt --dataset_name code

# Then train
python train_gpt2.py \
  --dataset=code \
  --n_layer=12 --n_head=8 --n_embd=512 \
  --max_iters=30000 --block_size=512 \
  2>&1 | tee out/training_code.log
```

### Resume Training
```bash
python train_gpt2.py \
  --init_from=resume \
  --max_iters=30000
```

---

## 🎯 Tips for Better Results

1. **More data is better** - Aim for at least 1MB of text
2. **Longer training** - Don't stop too early, watch validation loss
3. **Adjust temperature** - Lower (0.5-0.7) for consistent output, higher (0.9-1.2) for creativity
4. **Use dropout** - Especially for small datasets (<10MB)
5. **Monitor overfitting** - If val_loss >> train_loss, reduce model size or add dropout
6. **Be patient** - Good models take time (days, not hours)

---

## 🐛 Troubleshooting

**"Out of memory"**
- Reduce `--batch_size` to 4 or 8
- Reduce `--block_size` to 256 or 512
- Make model smaller

**"Loss not decreasing"**
- Train longer (more iterations)
- Increase learning rate: `--learning_rate=0.001`
- Check your data quality

**"Loss too low, but output is gibberish"**
- Model is overfitting
- Add dropout: `--dropout=0.2`
- Get more training data

**"Training too slow"**
- Reduce `--eval_interval` saves (slows down checkpointing)
- Use smaller `--eval_iters=50` (faster validation)
- Increase `--batch_size` if you have RAM
