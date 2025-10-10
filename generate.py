"""
Generate text using trained GPT model
Usage: python generate.py [--prompt "Your prompt here"] [--num_samples 5] [--max_tokens 200]
"""

import os
import pickle
import torch
from model import GPT

# -----------------------------------------------------------------------------
# Configuration
checkpoint_path = 'out/ckpt.pt'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
temperature = 0.8  # Higher = more random, lower = more predictable (0.0-1.0)
top_k = 200  # Consider only top K most likely tokens
num_samples = 5  # How many different samples to generate
max_new_tokens = 200  # How many tokens to generate per sample
seed = 1337
# -----------------------------------------------------------------------------

print("=" * 70)
print("GPT-2 Text Generator")
print("=" * 70)

# Load checkpoint
print(f"\n📂 Loading model from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model_args = checkpoint['model_args']

# Load tokenizer
meta_path = 'data/shakespeare/meta.pkl'
print(f"📖 Loading tokenizer from: {meta_path}")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Create model
print(f"🧠 Creating model ({checkpoint['iter_num']:,} iterations trained)")
from model import GPTConfig
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

model.eval()
model.to(device)

print(f"✅ Model loaded successfully!")
print(f"   - Device: {device}")
print(f"   - Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print(f"   - Training iterations: {checkpoint['iter_num']:,}")
print(f"   - Best validation loss: {checkpoint['best_val_loss']:.4f}")

# Generate text
torch.manual_seed(seed)
print("\n" + "=" * 70)
print("GENERATING SHAKESPEARE-LIKE TEXT")
print("=" * 70)

# Default prompts to try
prompts = [
    "\n",  # Start from newline (let model be creative)
    "ROMEO:",
    "JULIET:",
    "To be or not to be",
    "First Citizen:\n",
]

for i, start in enumerate(prompts):
    print(f"\n{'─' * 70}")
    print(f"Sample {i+1}/{len(prompts)}")
    print(f"Prompt: {repr(start)}")
    print(f"{'─' * 70}")

    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        generated_text = decode(y[0].tolist())

    print(generated_text)
    print()

print("\n" + "=" * 70)
print("Generation complete!")
print("=" * 70)

# Interactive mode
print("\n💡 Want to try your own prompt? Run this:")
print("   python generate_interactive.py")
