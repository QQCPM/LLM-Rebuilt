"""
Data preparation script - Shakespeare dataset
Following Andrej Karpathy's educational approach for learning GPT.

This downloads the Shakespeare dataset and prepares it for training.
Run this first before training: python prepare_shakespeare.py
"""

import os
import requests
import numpy as np
import pickle

# Download Shakespeare dataset
print("Downloading Shakespeare dataset...")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists('shakespeare.txt'):
    with open('shakespeare.txt', 'w') as f:
        f.write(requests.get(url).text)
    print("✅ Downloaded shakespeare.txt")
else:
    print("✅ shakespeare.txt already exists")

# Read the data
with open('shakespeare.txt', 'r') as f:
    data = f.read()

print(f"Length of dataset in characters: {len(data):,}")

# Get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} characters")
print("All characters:", ''.join(chars))

# Create character-to-integer mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Helper functions
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Test encoding/decoding
print("\nTesting encoding/decoding:")
test_str = "Hello world!"
encoded = encode(test_str)
decoded = decode(encoded)
print(f"Original: {test_str}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

# Encode entire dataset
data_encoded = encode(data)
print(f"\nTotal tokens: {len(data_encoded):,}")

# Split into train/val
n = len(data_encoded)
train_data = data_encoded[:int(n*0.9)]
val_data = data_encoded[int(n*0.9):]

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")

# Create data directory
os.makedirs('data/shakespeare', exist_ok=True)

# Save as binary files (compatible with train_gpt2.py)
train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)

train_ids.tofile('data/shakespeare/train.bin')
val_ids.tofile('data/shakespeare/val.bin')

print(f"✅ Saved train.bin: {len(train_ids):,} tokens")
print(f"✅ Saved val.bin: {len(val_ids):,} tokens")

# Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open('data/shakespeare/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print(f"✅ Saved meta.pkl with vocab_size={vocab_size}")

print("\n🎉 Data preparation complete!")
print("Now you can train with: python train_gpt2.py --dataset=shakespeare --n_layer=4 --n_head=4 --n_embd=128 --max_iters=1000")