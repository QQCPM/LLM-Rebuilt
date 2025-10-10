"""
Calculate GPT model size for different configurations
Helps you design models of specific parameter counts
"""

def calculate_params(n_layer, n_head, n_embd, vocab_size=50304, block_size=1024):
    """Calculate total trainable parameters for a GPT model"""

    # Token embeddings
    token_emb = vocab_size * n_embd

    # Position embeddings
    pos_emb = block_size * n_embd

    # Per-layer parameters (one transformer block)
    # LayerNorm 1: n_embd parameters (just weight, no bias in our config)
    ln1 = n_embd

    # Attention: c_attn (3 * n_embd * n_embd) + c_proj (n_embd * n_embd)
    attn_params = 3 * n_embd * n_embd + n_embd * n_embd  # = 4 * n_embd^2

    # LayerNorm 2: n_embd parameters
    ln2 = n_embd

    # MLP: c_fc (4*n_embd * n_embd) + c_proj (n_embd * 4*n_embd)
    mlp_params = 4 * n_embd * n_embd + n_embd * 4 * n_embd  # = 8 * n_embd^2

    # Total per layer
    per_layer = ln1 + attn_params + ln2 + mlp_params

    # Final layer norm
    final_ln = n_embd

    # Total (note: lm_head shares weights with token embeddings, so not counted)
    total = token_emb + pos_emb + (per_layer * n_layer) + final_ln

    # Non-embedding count (what's typically reported)
    non_emb = (per_layer * n_layer) + final_ln

    return {
        'total': total,
        'non_embedding': non_emb,
        'token_emb': token_emb,
        'pos_emb': pos_emb,
        'per_layer': per_layer,
        'layers': n_layer,
    }

def find_config_for_target(target_params_millions, vocab_size=50304):
    """Find configurations that get close to target parameter count"""

    target = target_params_millions * 1e6
    configs = []

    # Try different configurations
    for n_layer in [6, 8, 10, 12, 16, 20, 24]:
        for n_embd in [192, 256, 384, 512, 768, 1024]:
            # n_head must divide n_embd
            for n_head in [4, 6, 8, 12, 16]:
                if n_embd % n_head == 0:
                    params = calculate_params(n_layer, n_head, n_embd, vocab_size)
                    diff = abs(params['total'] - target)
                    configs.append({
                        'n_layer': n_layer,
                        'n_head': n_head,
                        'n_embd': n_embd,
                        'params_M': params['total'] / 1e6,
                        'non_emb_M': params['non_embedding'] / 1e6,
                        'diff': diff
                    })

    # Sort by how close to target
    configs.sort(key=lambda x: x['diff'])
    return configs[:5]

# Example configurations
print("=" * 80)
print("GPT MODEL SIZE CALCULATOR")
print("=" * 80)

print("\n📊 YOUR CURRENT MODEL:")
current = calculate_params(4, 4, 128, vocab_size=65, block_size=1024)
print(f"  Layers: 4, Heads: 4, Embedding: 128")
print(f"  Total params: {current['total'] / 1e6:.2f}M")
print(f"  Non-embedding: {current['non_embedding'] / 1e6:.2f}M")

print("\n" + "=" * 80)
print("SUGGESTED CONFIGURATIONS")
print("=" * 80)

targets = [1, 5, 10, 25, 50, 100]
for target in targets:
    print(f"\n🎯 TARGET: ~{target}M parameters")
    print("-" * 80)
    configs = find_config_for_target(target)

    for i, cfg in enumerate(configs[:3], 1):
        print(f"  {i}. layers={cfg['n_layer']:2d}, heads={cfg['n_head']:2d}, embd={cfg['n_embd']:4d} "
              f"→ {cfg['params_M']:6.2f}M total ({cfg['non_emb_M']:6.2f}M non-emb)")

print("\n" + "=" * 80)
print("FAMOUS MODEL SIZES")
print("=" * 80)

famous_models = [
    ("Your Current Model", 4, 4, 128, 65, 1024),
    ("Small GPT", 6, 6, 384, 50304, 1024),
    ("Medium GPT", 12, 12, 768, 50304, 1024),
    ("GPT-2 Small (124M)", 12, 12, 768, 50257, 1024),
    ("GPT-2 Medium (350M)", 24, 16, 1024, 50257, 1024),
    ("GPT-2 Large (774M)", 36, 20, 1280, 50257, 1024),
]

for name, n_l, n_h, n_e, vocab, block in famous_models:
    params = calculate_params(n_l, n_h, n_e, vocab, block)
    print(f"  {name:25s}: {params['total']/1e6:8.2f}M total, {params['non_embedding']/1e6:8.2f}M non-emb")

print("\n" + "=" * 80)
print("💡 USAGE")
print("=" * 80)
print("To train a model with specific size, use these flags:")
print("  python train_gpt2.py --n_layer=X --n_head=Y --n_embd=Z")
print("\nExample for ~10M parameters:")
print("  python train_gpt2.py --n_layer=12 --n_head=6 --n_embd=384")
