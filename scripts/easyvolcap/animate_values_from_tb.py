import os
import numpy as np
import matplotlib.pyplot as plt
from easyvolcap.utils.console_utils import *
from matplotlib.font_manager import FontProperties
from tensorboard.backend.event_processing import event_accumulator

# Configuration parameters
log_dir = 'data/record/easyvolcap'
output_dir = 'data/output'

# Font path and registration
font_path = 'assets/fonts/CascadiaCodePL-SemiBold.otf'
font_prop = FontProperties(fname=font_path)

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Extract scalar data from TensorBoard logs
ea = event_accumulator.EventAccumulator(log_dir, size_guidance={'scalars': 0})
ea.Reload()

loss_values = ea.Scalars('TRAIN/loss')
ssim_values = ea.Scalars('VAL/ssim_mean')
psnr_values = ea.Scalars('VAL/psnr_mean')
lpips_values = ea.Scalars('VAL/lpips_mean')

# Extract steps and corresponding values
steps = [s.step for s in ssim_values]
losses = [s.value for s in loss_values]
ssims = [s.value for s in ssim_values]
psnrs = [s.value for s in psnr_values]
lpipses = [s.value for s in lpips_values]

# Function for smoothing data


def smooth_data(data, weight=0.85):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# Set style properties for a dark background
plt.style.use('dark_background')

# Define a function to save plots


def save_plot(steps, values, ylabel, output_dir, prefix, jump, color):
    smoothed_values = smooth_data(values)
    for idx, i in enumerate(tqdm(range(0, len(steps), jump))):
        plt.figure(figsize=(8, 6), dpi=300)  # Set figure size and resolution
        plt.plot(steps[:i + 1], values[:i + 1], label=f'Original {ylabel}', color=color, alpha=0.3)  # Original data with reduced opacity
        plt.plot(steps[:i + 1], smoothed_values[:i + 1], label=f'Smoothed {ylabel}', color=color)
        plt.xlabel('Steps', fontproperties=font_prop, color='white')
        plt.ylabel(ylabel, fontproperties=font_prop, color='white')
        plt.title(f"{ylabel} at Step: {steps[i]}", fontproperties=font_prop, color='white')
        frame_path = os.path.join(output_dir, prefix, f"{idx:04d}.png")
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        plt.legend()
        plt.savefig(frame_path, facecolor='black')
        plt.close()


# Generate and save plots for each metric
colors = ['cyan', 'magenta', 'yellow', 'green']
metrics = [("SSIM", ssims, "SSIM"), ("PSNR", psnrs, "PSNR"), ("LPIPS", lpipses, "LPIPS"), ("LOSS", losses, "LOSS")]

for ylabel, values, prefix in metrics:
    color = colors.pop(0)
    save_plot(steps, values, ylabel, output_dir, prefix, jump=10, color=color)

print("Finished generating frames!")
