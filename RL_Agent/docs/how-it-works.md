# How It Works

## Startup (`main.py`)

1. `main.py` parses command-line arguments (mode, config path, device, log level).
2. Loads `config.yaml` which contains server info, PPO hyperparameters, reward values, and logging settings.
3. Creates output directories (`models/`, `logs/`) if they don't exist.
4. Detects whether a GPU is available and logs the device being used.
5. Creates a `PokeAgent` (`agent.py`) which manages the PPO model lifecycle.

## Self-Play Mode

### Environment Setup (`agent.py`, `environment.py`)

6. Two `SelfPlayOpponent` instances are created (`environment.py`). These are lightweight Python objects that provide a `choose_move()` method. They are the brain of the **opponent** — they decide what move to make — but they do **not** connect to the server themselves. Each starts by playing randomly, then periodically receives a frozen deep copy of the current PPO model so it gradually becomes a stronger opponent (see steps 21-22).
7. Six server accounts are used (all configured in `config.yaml`):
   - **RL_Agent_1** and **RL_Agent_2**: The RL learner's server accounts, one per parallel environment. These are the accounts whose perspective **PPO trains from**.
   - **InternalOpp_A** and **InternalOpp_B**: Server accounts for the opponent side in each environment. poke-env requires two real server-connected accounts to run a battle, so these exist as the opponent's **server presence** (the body on the server). However, their actual moves are **not** decided by anything on the server — they are decided locally by the `SelfPlayOpponent` object. They essentially act as dummy puppets that relay whatever `SelfPlayOpponent` decides.
   - **RL_Eval_1** and **RL_Eval_2**: Evaluation accounts that always make random moves. Reserved for the separate evaluation environment (see step 23).
8. Two `PokemonEnv` instances (subclass of poke-env's `SinglesEnv`) are created, each wrapped in a `SingleAgentWrapper` (`environment.py`). This converts the PettingZoo multi-agent environment into a standard Gymnasium single-agent environment that SB3 can use.
9. Each `SingleAgentWrapper` receives a `SelfPlayOpponent` as its opponent. When the wrapper calls `env.step()`, it asks the opponent's `choose_move()` for the second player's action and packages both actions together. **The opponent's actions do not feed into the PPO training buffer** — PPO only ever sees experience from RL_Agent_1/2's perspective. The opponent solely determines what the agent battles against, which in turn shapes the rewards RL_Agent_1/2 receives.
10. Both wrapped environments are placed into a `DummyVecEnv` (vectorized environment), so PPO collects experience from two battles simultaneously.

### Model Loading (`agent.py`)

11. If `models/poke_ppo.zip` exists, the saved model is loaded. Otherwise, a fresh `MaskablePPO` model is built with the configured hyperparameters (learning rate, network architecture, clip range, etc.).

### Observation (`observation.py`)

12. Each turn, the environment calls `embed_battle()` which converts the battle state into a **139-element float vector**:
    - Active Pokemon: HP, types (18-element multi-hot), status (one-hot), stat boosts.
    - Available moves (4 slots): base power, type effectiveness, category, accuracy, priority, PP fraction.
    - Opponent active Pokemon: same features as own active.
    - Team info: HP fractions, fainted flags, and switch availability for all 6 slots.
    - Opponent team: HP fractions and fainted flags for revealed Pokemon.
    - Field conditions: weather (one-hot, 8 slots) and terrain (one-hot, 5 slots).
    - Flags: whether Terastallization is available, and the normalized turn count.

### Action Masking (`environment.py`)

13. Before the model picks an action, an **action mask** is computed. The action space has 26 slots:
    - 0-5: Switch to team slot 0-5.
    - 6-9: Use move 0-3.
    - 10-25: Use move with a modifier (mega, z-move, dynamax, terastallize).
14. The mask is built by checking each action index against the battle's `valid_orders`. Only legal actions get a `True` in the mask.
15. `MaskablePPO` uses this mask to zero out the probability of illegal actions before sampling, so the agent never picks an invalid move.

### PPO Training Loop (`agent.py`)

16. PPO collects a rollout of 512 steps per environment (configurable via `n_steps`). With 2 environments, that's 1024 total steps per rollout.
17. After each rollout, PPO runs 10 gradient update epochs (configurable via `n_epochs`) on mini-batches of size 64.
18. The policy loss uses the PPO clipped surrogate objective (clip range 0.2). The value loss trains the value head. An entropy bonus (coefficient 0.005) encourages exploration.
19. Gradients are clipped to a max norm of 0.5.

### Reward (`environment.py`)

20. Rewards are computed as **deltas** each turn using `reward_computing_helper`:
    - +4.0 per opponent Pokemon fainted, -4.0 per own Pokemon fainted.
    - +1.0 per HP advantage (scaled by team HP fractions).
    - +0.3 for inflicting status conditions, -0.3 for receiving them.
    - +25.0 for winning, -25.0 for losing.
    - Each turn's reward is the change in this score since the last turn.

### Self-Play Opponent Updates (`utils.py`, `environment.py`)

21. Every 40,000 steps (configurable), `SelfPlayOpponentUpdateCallback` deep-copies the current PPO model into both `SelfPlayOpponent` instances (`utils.py`). Before the first update, opponents play **randomly** (no model is set).
22. After each update, the opponent holds a **frozen snapshot** of the PPO model from that moment in time — it will not change again until the next update. It uses this frozen copy with action masking to pick its moves. As training progresses, the opponents cycle through progressively stronger snapshots, giving the RL agent an increasingly difficult challenge. These frozen opponent actions still do **not** enter the PPO training buffer — they only influence what the agent faces, which shapes the rewards the agent receives.

### Evaluation and Checkpoints (`agent.py`, `utils.py`)

23. A dedicated evaluation environment is created using the **RL_Eval_1** and **RL_Eval_2** accounts. These accounts exist solely to host evaluation battles on the server without interfering with the training accounts (RL_Agent_1/2 and InternalOpp_A/B) which are busy with training battles. The eval environment's opponent is a permanently random `SelfPlayOpponent` — it is never updated and never receives the PPO model. Every 25,000 steps, `MaskableEvalCallback` runs 30 battles in this environment with the current PPO model acting deterministically. This measures absolute skill against a fixed random baseline. Evaluation results do **not** feed into PPO training — they are purely for monitoring progress.
24. Model checkpoints are saved every 50,000 steps. The best model found across all evaluations (by win rate against the random opponent) is saved separately to `models/best/`.
25. Battle metrics (win rate, average reward, battle length, throughput) are logged to TensorBoard and `logs/battle_metrics.csv` every 10 battles (`utils.py`).

### Training Completion (`agent.py`)

26. Training runs for 3,000,000 total timesteps (configurable, or overridden via `--timesteps`).
27. On completion (or Ctrl+C), the final model is saved to `models/poke_ppo.zip` and all environments are closed.

## VS Player Mode

28. A dummy environment is created temporarily using the **RL_Eval_1** and **RL_Eval_2** accounts just to initialize/load the PPO model — this keeps RL_Agent_1 free for the actual VS player connection (`agent.py`).
29. A `VsPlayerRunner` connects to the Showdown server under **RL_Agent_1** (`environment.py`). There is no opponent account — the human challenger connects from their own account.
30. The runner calls `accept_challenges(opponent=None, n_challenges=1)` in a loop, accepting any incoming challenge (`agent.py`).
31. During each battle, `choose_move()` runs the PPO model forward pass with action masking and stores the trajectory (observations, actions, log-probabilities, values) (`environment.py`).
32. When the battle ends, `_battle_finished_callback` computes Monte Carlo returns from the stored trajectory and runs an **episodic PPO update** — multiple gradient steps on the single episode using the clipped surrogate objective (`environment.py`).
33. Win/loss stats and losses are logged to TensorBoard under the `vsplayer/` prefix.
34. The model is periodically saved to disk. The loop continues until interrupted with Ctrl+C or the timestep limit is reached (`agent.py`).
