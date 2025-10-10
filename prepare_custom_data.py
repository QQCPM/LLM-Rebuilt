"""
Prepare custom datasets for GPT training
Supports: text files, code, conversations, any text data

Usage:
    python prepare_custom_data.py --input your_data.txt --dataset_name my_dataset
"""

import os
import argparse
import numpy as np
import pickle

def prepare_dataset(input_file, dataset_name, train_split=0.9):
    """
    Prepare any text file for GPT training

    Args:
        input_file: Path to your text file
        dataset_name: Name for the dataset (creates data/{dataset_name}/)
        train_split: Fraction of data for training (default 0.9 = 90%)
    """

    print("=" * 70)
    print(f"PREPARING DATASET: {dataset_name}")
    print("=" * 70)

    # Read the data
    print(f"\n📂 Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    print(f"✅ Loaded {len(data):,} characters")

    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    print(f"\n📖 Vocabulary:")
    print(f"   Size: {vocab_size} unique characters")
    if vocab_size <= 100:
        print(f"   Characters: {repr(''.join(chars))}")
    else:
        print(f"   Sample: {repr(''.join(chars[:50]))}... (showing first 50)")

    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode the data
    print(f"\n🔢 Encoding data...")
    encoded = [stoi[c] for c in data]
    print(f"✅ Encoded {len(encoded):,} tokens")

    # Split into train/val
    n = len(encoded)
    train_size = int(n * train_split)
    train_data = encoded[:train_size]
    val_data = encoded[train_size:]

    print(f"\n📊 Split:")
    print(f"   Training: {len(train_data):,} tokens ({len(train_data)/n*100:.1f}%)")
    print(f"   Validation: {len(val_data):,} tokens ({len(val_data)/n*100:.1f}%)")

    # Create output directory
    output_dir = f'data/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Save as binary files
    train_ids = np.array(train_data, dtype=np.uint16)
    val_ids = np.array(val_data, dtype=np.uint16)

    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))

    print(f"\n💾 Saved binary files:")
    print(f"   {output_dir}/train.bin: {len(train_ids):,} tokens")
    print(f"   {output_dir}/val.bin: {len(val_ids):,} tokens")

    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }

    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"   {output_dir}/meta.pkl: vocab_size={vocab_size}")

    print("\n" + "=" * 70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 70)

    print(f"\n🚀 To train on this dataset, run:")
    print(f"   python train_gpt2.py \\")
    print(f"     --dataset={dataset_name} \\")
    print(f"     --n_layer=12 --n_head=6 --n_embd=384 \\")
    print(f"     --max_iters=10000")

    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare custom dataset for GPT training')
    parser.add_argument('--input', type=str, required=True, help='Input text file')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name for the dataset')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train/val split (default 0.9)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        exit(1)

    prepare_dataset(args.input, args.dataset_name, args.train_split)
