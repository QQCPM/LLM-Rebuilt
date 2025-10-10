"""
Interactive text generation with your trained GPT model
"""

import os
import pickle
import torch
from model import GPT

# Setup
checkpoint_path = 'out/ckpt.pt'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print("Loading model...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load tokenizer
with open('data/shakespeare/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Create model
from model import GPTConfig
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

print(f"✅ Model loaded! ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters)")
print(f"   Trained for {checkpoint['iter_num']:,} iterations")
print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
print()

# Interactive loop
while True:
    print("─" * 70)
    prompt = input("Enter your prompt (or 'quit' to exit): ")

    if prompt.lower() in ['quit', 'exit', 'q']:
        print("Goodbye! 👋")
        break

    # Get generation parameters
    try:
        max_tokens = int(input("Max tokens to generate (default 200): ") or "200")
        temperature = float(input("Temperature 0.0-1.0 (default 0.8): ") or "0.8")
    except ValueError:
        print("Invalid input, using defaults")
        max_tokens = 200
        temperature = 0.8

    # Generate
    print(f"\n🎭 Generating text...\n")

    # Check if prompt contains unknown characters
    unknown_chars = [c for c in prompt if c not in stoi]
    if unknown_chars:
        print(f"❌ Error: Your prompt contains characters not in Shakespeare's vocabulary:")
        print(f"   Unknown characters: {set(unknown_chars)}")
        print(f"\n💡 This model only knows these characters:")
        print(f"   {repr(''.join(itos.values()))}")
        print(f"\n   Try prompts like: 'ROMEO:', 'To be', 'My lord', etc.")
        continue

    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(x, max_tokens, temperature=temperature, top_k=200)
        text = decode(y[0].tolist())

    print("─" * 70)
    print(text)
    print("─" * 70)
    print()
