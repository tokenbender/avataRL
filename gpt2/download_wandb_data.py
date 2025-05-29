#!/usr/bin/env python
"""
Download data from WandB runs
"""

import wandb
import pandas as pd
import json

# Initialize wandb API
api = wandb.Api()

# Get your run
# Replace with your actual project and run ID
PROJECT = "gpt2-grpo-guided-exhaustive"  # or whatever project name you used
RUN_ID = "latest"  # or specific run ID

# Get the run
runs = api.runs(f"{api.default_entity}/{PROJECT}")
if RUN_ID == "latest":
    run = runs[0]  # Most recent run
else:
    run = api.run(f"{api.default_entity}/{PROJECT}/{RUN_ID}")

print(f"Downloading data from run: {run.name} ({run.id})")

# Download run history (all logged metrics)
history = run.history()
history.to_csv("run_metrics.csv")
print(f"Saved metrics to run_metrics.csv ({len(history)} steps)")

# Download config
config = json.dumps(dict(run.config), indent=2)
with open("run_config.json", "w") as f:
    f.write(config)
print("Saved config to run_config.json")

# Download summary metrics
summary = json.dumps(dict(run.summary), indent=2)
with open("run_summary.json", "w") as f:
    f.write(summary)
print("Saved summary to run_summary.json")

# Download any files saved to wandb
print("\nFiles in run:")
for file in run.files():
    print(f"  {file.name}")
    # Uncomment to download:
    # file.download()

# Get specific metrics
print(f"\nFinal metrics:")
print(f"  Final reward: {run.summary.get('reward', 'N/A')}")
print(f"  Final KL: {run.summary.get('kl', 'N/A')}")
print(f"  Total chars: {run.summary.get('chars', 'N/A')}")

# Plot learning curves
if len(history) > 0:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Rewards
    axes[0, 0].plot(history['chars'], history['reward'])
    axes[0, 0].set_title('Reward over time')
    axes[0, 0].set_xlabel('Characters seen')
    
    # KL divergence
    axes[0, 1].plot(history['chars'], history['kl'])
    axes[0, 1].set_title('KL Divergence')
    axes[0, 1].set_xlabel('Characters seen')
    
    # Entropy
    axes[1, 0].plot(history['chars'], history['entropy'])
    axes[1, 0].set_title('Policy Entropy')
    axes[1, 0].set_xlabel('Characters seen')
    
    # Loss
    axes[1, 1].plot(history['chars'], history['pol_loss'])
    axes[1, 1].set_title('Policy Loss')
    axes[1, 1].set_xlabel('Characters seen')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("\nSaved learning curves to learning_curves.png")