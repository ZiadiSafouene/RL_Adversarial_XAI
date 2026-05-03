# RL-Attacker: Reinforcement Learning for Adversarial Attacks and XAI

Project for generating adversarial attacks using Reinforcement Learning and interpreting the results using XAI metrics and clustering.

## Project Structure

```
├── data/                   # Raw and processed datasets (e.g., CIFAR-10)
├── notebooks/              # Exploratory data analysis and visualization scratchpads
├── results/                # TensorBoard logs, saved model weights, and generated plots
│
├── rl_attacker/            # Main Python package
│   ├── __init__.py
│   ├── target_models/      # Code for the victim CNNs
│   │   ├── train_target.py # Script to train baseline model
│   │   └── architectures.py# ResNet or custom CNN definitions
│   │
│   ├── baselines/          # Standard attacks for comparison
│   │   └── pgd_fgsm.py     # Torchattacks or CleverHans implementations
│   │
│   ├── environment/        # The custom RL MDP
│   │   ├── __init__.py
│   │   └── adv_env.py      # Gymnasium Env class (States, Actions, Rewards)
│   │
│   ├── agents/             # RL agent training and evaluation
│   │   ├── train_ppo.py    # Training loop for PPO
│   │   └── evaluate.py     # Script to run policy and collect trajectories
│   │
│   └── analysis/           # Phase 5: XAI and Clustering
│       ├── clustering.py   # KMeans/DBSCAN on trajectories
│       ├── visualizer.py   # Overlaying patches on images
│       └── xai_metrics.py  # Integrated Gradients and IoU calculations
│
├── requirements.txt        # Dependencies (PyTorch, Gymnasium, Stable-Baselines3, Captum)
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
