# How to Run

## Navigate To Corrrect Directory

``` bash
cd RL_Agent
```

Ensures you are in the RL subdirectory.

## Install Dependencies

```bash
pip install -r requirements.txt
```

Installs all required Python packages (poke-env, stable-baselines3, PyTorch, etc.).

## Self-Play Training

```bash
python main.py selfplay
```

Trains the agent against a copy of itself using two parallel battle environments. The opponent starts random and is periodically updated to match the agent's current skill level.

## Self-Play Training (Custom Timesteps)

```bash
python main.py selfplay --timesteps 100000
```

Same as above but overrides the total training steps from config.yaml. Useful for quick test runs.

## VS Player Mode

```bash
python main.py vs_player
```

Connects to the Showdown server and waits for incoming battle challenges. The agent plays against human opponents and learns from each battle. Challenge the bot's username (configured in config.yaml) on the server to start a match.

## Optional Flags

```bash
python main.py selfplay --config custom.yaml
```

Use a different configuration file instead of the default config.yaml.

```bash
python main.py selfplay --device cpu
```

Force CPU training instead of auto-detecting GPU. Options: `auto`, `cuda`, `cpu`.

```bash
python main.py selfplay --log-level DEBUG
```

Set logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

## View Training Metrics

```bash
tensorboard --logdir logs
```

Opens a TensorBoard dashboard showing win rate, reward, loss curves, and other training metrics.

## Delete Training Progress And Metrics

Delete the following directories:
* `logs/` - TensorBoard logs + battle_metrics.csv (progress tracking).
* `models/` - All saved model weights.
