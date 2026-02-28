# Project Layout

```
Pokemon_Showdown_RL_Agent/
├── main.py
├── config.yaml
├── requirements.txt
├── src/
│   ├── observation.py
│   ├── environment.py
│   ├── agent.py
│   └── utils.py
├── models/
│   ├── poke_ppo.zip
│   └── best/
├── logs/
│   ├── training.log
│   ├── battle_metrics.csv
│   ├── selfplay_0/
│   └── vsplayer/
└── docs/
    ├── how-to-run.md
    ├── brief-overview.md
    ├── how-it-works.md
    └── layout.md
```

## Files

| File | Description |
|---|---|
| `main.py` | Entry point. Parses CLI arguments, loads config, creates the agent, and launches the selected training mode. |
| `config.yaml` | All configuration: server URL/usernames, PPO hyperparameters, reward values, network architecture, and logging settings. |
| `requirements.txt` | Python package dependencies. |

## src/

| File | Description |
|---|---|
| `observation.py` | Converts a battle state into a 139-element float vector (the observation). Encodes Pokemon stats, moves, types, weather, terrain, etc. |
| `environment.py` | Defines `PokemonEnv` (the RL environment), `SelfPlayOpponent` (local opponent for self-play), `VsPlayerRunner` (server-connected player for human battles), and factory functions for creating environments. Also contains the action masking logic. |
| `agent.py` | `PokeAgent` class that manages the MaskablePPO model — building, loading, saving, and running both training modes. |
| `utils.py` | Logging setup and SB3 callbacks: `BattleMetricsCallback` (tracks win rate, rewards, writes CSV) and `SelfPlayOpponentUpdateCallback` (copies current policy to frozen opponents). |

## models/

| Path | Description |
|---|---|
| `models/poke_ppo.zip` | The saved PPO model checkpoint. Loaded on startup and saved after training. |
| `models/best/` | Stores the best-performing model found during evaluation against a random opponent. |

## logs/

| Path | Description |
|---|---|
| `logs/training.log` | Text log of training events, battle results, and status messages. |
| `logs/battle_metrics.csv` | CSV with per-checkpoint stats: timestep, battles played, wins, losses, win rate, average reward, average battle length, and throughput. |
| `logs/selfplay_0/` | TensorBoard event files for self-play training (PPO losses, rewards, etc.). |
| `logs/vsplayer/` | TensorBoard event files for VS player mode (win rate, episode returns, losses). |
