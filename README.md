# Multi Technique RL Compression for LLMs

This repository contains a  reinforcement-learning setup for generating compression plans across model layers. The implementation includes:

- `Environment.py` — layer definitions, action aggressiveness, and PPL (perplexity) hit computation.
- `CompressionEnvironment.py` — a Gym-compatible wrapper around `Environment`.
- `DQNAgent.py` — a DQN-style agent implementation using PyTorch.
- `Qnetwork.py` — feed-forward Q-network.
- `ReplayBuffer.py` — replay buffer implementation.
- `TrainAgent.py` — training loop and a small inference script at the bottom.

## Requirements

- Python 3.8+ (3.11 recommended)
- PyTorch
- Gym
- NumPy

Create a virtual environment and install dependencies. Example using `venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
pip install --upgrade pip
pip install torch gym numpy
```

If you already have a conda environment you can create and activate one instead:

```powershell
conda create -n rl_compress python=3.11 -y
conda activate rl_compress
pip install torch gym numpy
```

Note: Install the appropriate `torch` wheel for your platform/GPU following https://pytorch.org/get-started/locally/.

## Running training

The main training entrypoint is `TrainAgent.py`. Run it from the project root:

```powershell
python TrainAgent.py
```

This will:

- Instantiate `Environment` and wrap it with `CompressionEnvironment`.
- Create a `DQNAgent` and run training for 500 episodes (default in file).
- After training, run a short greedy inference pass for several PPL budgets and print the chosen actions per layer.

To change hyperparameters (learning rate, batch size, episodes, etc.), edit the `agent` and `trainer` construction near the bottom of `TrainAgent.py`.

## Project structure

- `Environment.py` — base environment definitions and `PPL_hit` logic.
- `CompressionEnvironment.py` — Gym wrapper, exposes `reset()` and `step()`.
- `DQNAgent.py` — agent API: `act()`, `remember()`, `learn()`.
- `Qnetwork.py` — PyTorch model definition.
- `ReplayBuffer.py` — experience replay storage.
- `TrainAgent.py` — trainer class and runnable script.

## Tips & Troubleshooting

- If training doesn't start due to missing packages, reinstall dependencies in the active environment.
- If GPU is desired, install a CUDA-enabled `torch` build and ensure CUDA drivers are set up.
- If you want deterministic runs, set seeds in each module and disable CUDA nondeterminism in PyTorch.
