# 🚀 Getting Started with Educational GPT-2

This is a simplified, educational implementation of GPT-2 following Andrej Karpathy's approach. Perfect for learning how language models work!

## 📁 Project Structure (Just 4 Files!)
```
├── model.py              # Complete GPT-2 implementation (~300 lines)
├── train_gpt2.py         # Training script (~330 lines)
├── prepare_shakespeare.py # Data preparation (~70 lines)
└── configurator.py       # Configuration overrides (~40 lines)
```

## 🏃 Quick Start

### 1. Prepare Data
```bash
python prepare_shakespeare.py
```
This downloads the Shakespeare dataset and prepares training/validation files.

### 2. Train a Small Model (Fast)
```bash
python train_gpt2.py --n_layer=4 --n_head=4 --n_embd=128 --max_iters=1000 --eval_interval=100
```

### 3. Train a Larger Model (Better Results)
```bash
python train_gpt2.py --n_layer=6 --n_head=6 --n_embd=384 --max_iters=5000 --eval_interval=500
```

### 4. Generate Text
```python
import torch
import pickle
from model import GPT, GPTConfig

# Load trained model
checkpoint = torch.load('out/ckpt.pt', map_location='cpu')
model = GPT(GPTConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.eval()

# Load tokenizer
with open('data/shakespeare/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Generate
prompt = "ROMEO:"
context = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
with torch.no_grad():
    generated = model.generate(context, max_new_tokens=200, temperature=0.8)
    print(decode(generated[0].tolist()))
```

## ⚙️ Configuration Options

You can override any hyperparameter via command line:
```bash
python train_gpt2.py --batch_size=32 --learning_rate=3e-4 --block_size=256
```

Key parameters:
- `n_layer`: Number of transformer layers (default: 12)
- `n_head`: Number of attention heads (default: 12)
- `n_embd`: Embedding dimension (default: 768)
- `block_size`: Context length (default: 1024)
- `batch_size`: Batch size (default: 12)
- `max_iters`: Training iterations (default: 600000)
- `learning_rate`: Learning rate (default: 6e-4)

## 🎯 Learning Goals

This implementation teaches you:
1. **Transformer Architecture**: Self-attention, feed-forward networks, layer normalization
2. **Training Loop**: Gradient accumulation, learning rate scheduling, checkpointing
3. **Text Generation**: Autoregressive sampling with temperature and top-k
4. **Optimization**: Weight tying, parameter initialization, AdamW optimizer

## 📚 Educational Features

- **Simple & Clean**: Each file is ~300 lines, easy to understand
- **Well Commented**: Clear explanations of key concepts
- **Hackable**: Modify any part to experiment and learn
- **Fast Training**: Works on CPU, optimized for GPU/MPS
- **Real Results**: Actually generates coherent text after training

## 🔧 Hardware Requirements

- **Minimum**: CPU only (slow but works)
- **Recommended**: GPU with 4GB+ VRAM
- **Optimal**: GPU with 8GB+ VRAM for larger models

## 📖 Next Steps

1. **Experiment**: Try different architectures and hyperparameters
2. **Datasets**: Use your own text data with prepare_shakespeare.py as template
3. **Analysis**: Study attention patterns and layer activations
4. **Scaling**: Train larger models with more data

Happy learning! 🎓