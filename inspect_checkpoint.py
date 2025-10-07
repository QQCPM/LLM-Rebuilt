"""
Inspect checkpoint file - see what's inside without loading the full model
"""
import torch
import os

checkpoint_path = 'out/ckpt.pt'

if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found at {checkpoint_path}")
    exit(1)

print(f"📂 Loading checkpoint from: {checkpoint_path}")
print(f"📊 File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB\n")

# Load checkpoint (CPU only, no GPU needed for inspection)
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=" * 60)
print("CHECKPOINT CONTENTS")
print("=" * 60)

# Show top-level keys
print("\n📋 Top-level keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Show model configuration
print("\n⚙️  Model Configuration:")
if 'model_args' in checkpoint:
    for key, value in checkpoint['model_args'].items():
        print(f"  {key}: {value}")

# Show training state
print("\n📈 Training State:")
if 'iter_num' in checkpoint:
    print(f"  Iteration: {checkpoint['iter_num']:,}")
if 'best_val_loss' in checkpoint:
    print(f"  Best Validation Loss: {checkpoint['best_val_loss']:.4f}")

# Show config (training hyperparameters)
print("\n🔧 Training Config:")
if 'config' in checkpoint:
    for key, value in sorted(checkpoint['config'].items()):
        print(f"  {key}: {value}")

# Model state info
print("\n🧠 Model State Dictionary:")
if 'model' in checkpoint:
    state_dict = checkpoint['model']
    print(f"  Total parameters: {len(state_dict)} tensors")
    print(f"\n  First 10 parameter names:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        tensor = state_dict[key]
        print(f"    {i+1}. {key}: shape={list(tensor.shape)}")

    # Calculate total parameter count
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\n  Total parameter count: {total_params:,} ({total_params/1e6:.2f}M)")

# Optimizer state info
print("\n🎯 Optimizer State:")
if 'optimizer' in checkpoint:
    opt_state = checkpoint['optimizer']
    print(f"  State keys: {list(opt_state.keys())}")
    if 'param_groups' in opt_state:
        print(f"  Number of param groups: {len(opt_state['param_groups'])}")
        print(f"  Learning rate: {opt_state['param_groups'][0]['lr']}")

print("\n" + "=" * 60)
print("✅ Checkpoint inspection complete!")
print("=" * 60)
