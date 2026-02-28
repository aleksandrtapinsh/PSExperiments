# Brief Overview

Reinforcement learning agent that plays Pokemon Showdown Random Battles (Gen 9). It learns to play through self-play and through battling human opponents on a private Showdown server.

## What It Does

The agent observes the current battle state (HP, types, moves, weather, etc.), picks an action (use a move or switch Pokemon), and receives a reward based on how well it's doing. Over many battles, it learns a policy that maximizes its win rate.

## How It Learns

The agent uses **PPO (Proximal Policy Optimization)** with **action masking** to ensure it only picks legal moves. It has two training modes:

- **Self-play**: The agent battles a frozen copy of itself, which is periodically updated. Two parallel environments run simultaneously for faster data collection.
- **VS Player**: The agent accepts challenges from human players on the server and learns from each battle in real time.

## Libraries Used

| Library | Purpose |
|---|---|
| **poke-env** (0.11.x) | Python interface to Pokemon Showdown; handles battle communication, game state parsing, and the RL environment |
| **Stable Baselines3** + **sb3-contrib** | PPO implementation and action masking (MaskablePPO) |
| **PyTorch** | Neural network backend for the policy and value networks |
| **Gymnasium** | Standard RL environment interface |
| **TensorBoard** | Training metric visualization |
| **NumPy** | Numerical operations for the observation embedding |
| **PyYAML** | Configuration file parsing |
| **Pandas** | CSV logging of battle statistics |
