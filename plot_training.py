"""
Plot training loss curves from log file
"""

import re
import matplotlib.pyplot as plt

log_file = 'out/training_20251006_231045.log'

# Extract iteration and loss from log
iterations = []
losses = []

with open(log_file, 'r') as f:
    for line in f:
        # Match lines like: "iter 123: loss 2.3456, time..."
        match = re.match(r'iter (\d+): loss ([\d.]+)', line)
        if match:
            iterations.append(int(match.group(1)))
            losses.append(float(match.group(2)))

# Extract validation losses
val_iterations = []
val_losses = []

with open(log_file, 'r') as f:
    for line in f:
        # Match lines like: "step 500: train loss 1.23, val loss 4.56"
        match = re.match(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
        if match:
            val_iterations.append(int(match.group(1)))
            val_losses.append(float(match.group(3)))

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(iterations, losses, alpha=0.6, label='Training Loss')
plt.plot(val_iterations, val_losses, 'ro-', label='Validation Loss', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Plot smoothed version
window = 50
smoothed = [sum(losses[max(0, i-window):i+1]) / len(losses[max(0, i-window):i+1])
            for i in range(len(losses))]
plt.plot(iterations, smoothed, label='Smoothed Training Loss')
plt.plot(val_iterations, val_losses, 'ro-', label='Validation Loss', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Smoothed Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('out/training_curves.png', dpi=150)
print(f"✅ Plot saved to: out/training_curves.png")
print(f"   Total iterations: {len(iterations)}")
print(f"   Initial loss: {losses[0]:.4f}")
print(f"   Final loss: {losses[-1]:.4f}")
print(f"   Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
print(f"   Final val loss: {val_losses[-1]:.4f}")

plt.show()
