# GPT-2 Rebuild:

This project implements GPT-2 (124M) from scratch following Andrej Karpathy's comprehensive 4-hour tutorial. The implementation focuses on understanding the deep technical details of modern language model training.

## Project Structure

```
LLM-Rebuild/
├── src/
│   ├── model/           # GPT-2 architecture implementation
│   │   ├── gpt2.py     # Main model definition
│   │   ├── attention.py # Attention mechanisms
│   │   └── layers.py   # Transformer layers
│   ├── training/        # Training components
│   │   ├── trainer.py  # Training loop
│   │   └── config.py   # Configuration
│   ├── optimization/    # Performance optimizations
│   │   └── gpu_utils.py # GPU and memory optimizations
│   ├── data/           # Data processing
│   │   └── dataset.py  # Dataset handling
│   └── evaluation/     # Model evaluation
│       └── hellaswag.py # HellaSwag benchmark
├── configs/            # Configuration files
├── checkpoints/        # Model checkpoints
├── notebooks/          # Jupyter notebooks for analysis
├── tests/             # Unit tests
├── scripts/           # Utility scripts
└── docs/              # Documentation
```

## Tutorial Sections

### Section 1: Architecture Implementation (0:00 - 1:22)
- **Part A**: Model setup, OpenAI weights loading, Hugging Face integration
- **Part B**: GPT-2 specific architecture (pre-norm, GELU, weight tying)

### Section 2: Hardware Optimization (1:22 - 2:15)
- **Part A**: GPU optimization (TF32, mixed precision BF16)
- **Part B**: Memory efficiency (torch.compile, Flash Attention)

### Section 3: Training at Scale (2:15 - 4:00)
- **Part A**: Training setup (learning rate scheduling, AdamW, gradient accumulation)
- **Part B**: Training dynamics (loss monitoring, gradient clipping, checkpointing)
- **Part C**: Advanced techniques (distributed training, evaluation)

## Key Features

- **GPT-2 (124M) implementation** matching OpenAI's architecture exactly
- **Hardware optimizations** for modern GPU training
- **Production-ready training pipeline** with monitoring and checkpointing
- **Comprehensive evaluation** including HellaSwag benchmark
- **Educational focus** with detailed documentation of each component

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training:
   ```bash
   python train.py
   ```

3. Evaluate model:
   ```bash
   python evaluate.py
   ```

## Technical Highlights

- Pre-norm instead of post-norm LayerNorm
- GELU activation with OpenAI's approximation
- Weight tying between embedding and output layers
- Vocabulary optimization (50,257 → 50,304 for "nice numbers")
- Flash Attention for memory-efficient computation
- Mixed precision training with BF16
- Cosine learning rate scheduling with warmup

## Based On

- [Andrej Karpathy's 4-hour GPT-2 tutorial](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [nanoGPT repository](https://github.com/karpathy/nanoGPT)
- GPT-2 and GPT-3 papers for hyperparameters
