# Gomoku AI (AlphaZero-style)

A Gomoku (Five-in-a-Row) AI combining **Monte Carlo Tree Search (MCTS)** with a **Residual CNN**, inspired by AlphaGo Zero. Learns entirely through self-play.

## Files

| File | Purpose |
|------|---------|
| `game.py` | Board logic, move validation, win detection (5 in a row), display |
| `mcts.py` | MCTS with PUCT selection, Dirichlet noise, temperature-based move sampling |
| `model.py` | 8-block ResNet with dual policy & value heads, training utilities |
| `train.py` | Multi-process self-play pipeline with replay buffer & data augmentation |
| `play.py` | Human vs. AI interface |

## Key Config (`train.py` → `Config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_playout` | 480 | MCTS simulations per move |
| `c_puct` | 5.2 | Exploration constant |
| `batch_size` | 256 | Training batch size |
| `learn_rate` | 2e-4 | Learning rate |
| `num_workers` | auto | Self-play workers |

## Requirements

Python 3.7+, PyTorch, NumPy
