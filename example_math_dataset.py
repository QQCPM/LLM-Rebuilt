#!/usr/bin/env python3
"""
Create a simple math dataset for training GPT
This teaches the model to do arithmetic!
"""

import random
import os

def generate_math_problems(num_problems=50000):
    """Generate simple arithmetic problems"""

    problems = []

    for _ in range(num_problems):
        a = random.randint(0, 99)
        b = random.randint(0, 99)

        # Addition
        problems.append(f"{a} + {b} = {a + b}")

        # Subtraction
        if a >= b:
            problems.append(f"{a} - {b} = {a - b}")

        # Multiplication (smaller numbers)
        if a <= 12 and b <= 12:
            problems.append(f"{a} * {b} = {a * b}")

    # Shuffle
    random.shuffle(problems)

    return problems

print("📊 Generating math dataset...")
problems = generate_math_problems(50000)

# Save to file
output_file = 'math_dataset.txt'
with open(output_file, 'w') as f:
    for problem in problems:
        f.write(problem + '\n')

print(f"✅ Created {len(problems):,} math problems")
print(f"📁 Saved to: {output_file}")
print(f"📏 File size: {os.path.getsize(output_file) / 1024:.1f} KB")
print()
print("Sample problems:")
for i in range(10):
    print(f"  {problems[i]}")
print()
print("🚀 Next steps:")
print("  1. Prepare dataset:")
print(f"     python prepare_custom_data.py --input {output_file} --dataset_name math")
print()
print("  2. Train model:")
print("     python train_gpt2.py --dataset=math --n_layer=8 --n_head=4 --n_embd=256 --max_iters=10000")
print()
print("  3. Test with prompts like:")
print('     "7 + 3 = " → model should complete with "10"')
