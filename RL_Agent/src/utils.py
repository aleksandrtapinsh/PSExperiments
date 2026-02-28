"""
utils.py
========
Logging helpers and SB3 custom callbacks.

Provides
--------
BattleMetricsCallback
    Logs win/loss rate, average episode reward, and a running CSV log to disk
    and TensorBoard after every completed battle.

SelfPlayOpponentUpdateCallback
    Periodically copies the current PPO policy into the frozen self-play
    opponents so they become progressively stronger.

setup_logging
    Configures Python logging with rich formatting for the console.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from .agent import PokeAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Console / file logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Configure root logger with console and file handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "training.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: List[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]
    for h in handlers:
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logging.basicConfig(level=level, handlers=handlers)
    # Quiet noisy third-party loggers
    for lib in ("websockets", "asyncio", "poke_env"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Battle metrics callback
# ---------------------------------------------------------------------------

class BattleMetricsCallback(BaseCallback):
    """
    Logs battle-level metrics to TensorBoard and a CSV file.

    Metrics tracked
    ---------------
    win_rate            Fraction of completed battles won (rolling 100-battle window)
    avg_episode_reward  Average undiscounted return per battle
    battles_per_minute  Battle throughput
    avg_battle_length   Average turns per battle (if reported by Monitor)
    """

    CSV_HEADER = [
        "timestep",
        "battles_total",
        "wins",
        "losses",
        "ties",
        "win_rate",
        "avg_reward",
        "avg_ep_len",
        "battles_per_min",
    ]

    def __init__(
        self,
        log_dir: str = "logs",
        battle_log_freq: int = 10,
        window: int = 100,
        verbose: int = 0,
    ) -> None:
        """Initialise rolling stat buffers, CSV path, and counters."""
        super().__init__(verbose=verbose)
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._battle_log_freq = battle_log_freq
        self._window = window

        # Rolling statistics
        self._outcomes: deque[int] = deque(maxlen=window)  # 1=win, 0=draw, -1=loss
        self._ep_rewards: deque[float] = deque(maxlen=window)
        self._ep_lens: deque[int] = deque(maxlen=window)

        self._total_battles = 0
        self._wins = 0
        self._losses = 0
        self._ties = 0
        self._start_time = time.time()

        self._csv_path = self._log_dir / "battle_metrics.csv"
        self._csv_initialized = False

    def _init_csv(self) -> None:
        """Write the CSV header row and mark the file as initialised."""
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADER)
        self._csv_initialized = True

    def _write_csv_row(self, row: List[Any]) -> None:
        """Append a data row to the CSV file, initialising it first if needed."""
        if not self._csv_initialized:
            self._init_csv()
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _on_step(self) -> bool:
        """Called every env step; extracts episode info from Monitor and logs metrics."""
        # SB3 Monitor wraps each env and populates `info` with episode stats
        # when an episode terminates (keys: "episode", "r", "l").
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                reward = float(ep.get("r", 0.0))
                length = int(ep.get("l", 0))

                self._ep_rewards.append(reward)
                self._ep_lens.append(length)
                self._total_battles += 1

                # Win/loss is embedded in reward sign for Pokémon
                # The Monitor info may also carry a "won" flag if set by env
                won = info.get("won", None)
                if won is True:
                    self._wins += 1
                    self._outcomes.append(1)
                elif won is False:
                    self._losses += 1
                    self._outcomes.append(-1)
                else:
                    # Infer from terminal reward heuristic
                    # (large positive → win, large negative → loss)
                    if reward > 10.0:
                        self._wins += 1
                        self._outcomes.append(1)
                    elif reward < -10.0:
                        self._losses += 1
                        self._outcomes.append(-1)
                    else:
                        self._ties += 1
                        self._outcomes.append(0)

                # Log to TensorBoard every battle_log_freq battles
                if self._total_battles % self._battle_log_freq == 0:
                    self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Write rolling win rate, reward, and throughput to TensorBoard and CSV."""
        if not self._outcomes:
            return

        wins_arr = np.array(self._outcomes)
        win_rate = float((wins_arr == 1).mean())
        avg_reward = float(np.mean(self._ep_rewards)) if self._ep_rewards else 0.0
        avg_len = float(np.mean(self._ep_lens)) if self._ep_lens else 0.0

        elapsed = time.time() - self._start_time
        battles_per_min = self._total_battles / (elapsed / 60.0) if elapsed > 0 else 0.0

        # TensorBoard
        self.logger.record("battle/win_rate", win_rate)
        self.logger.record("battle/avg_episode_reward", avg_reward)
        self.logger.record("battle/avg_battle_length", avg_len)
        self.logger.record("battle/battles_per_minute", battles_per_min)
        self.logger.record("battle/total_battles", self._total_battles)
        self.logger.dump(step=self.num_timesteps)

        # CSV
        self._write_csv_row([
            self.num_timesteps,
            self._total_battles,
            self._wins,
            self._losses,
            self._ties,
            round(win_rate, 4),
            round(avg_reward, 4),
            round(avg_len, 1),
            round(battles_per_min, 2),
        ])

        logger.info(
            f"[Step {self.num_timesteps:>8}] Battles: {self._total_battles} | "
            f"Win%: {win_rate:.1%} | AvgR: {avg_reward:+.2f} | "
            f"AvgLen: {avg_len:.1f} turns"
        )


# ---------------------------------------------------------------------------
# Self-play opponent update callback
# ---------------------------------------------------------------------------

class SelfPlayOpponentUpdateCallback(BaseCallback):
    """
    Copies the current PPO policy into the frozen self-play opponents every
    `update_freq` timesteps.

    Parameters
    ----------
    opponents : list
        List of SelfPlayOpponent (or RandomPlayer) instances used as env
        opponents in the VecEnv.
    update_freq : int
        How often (in total env steps) to refresh opponent weights.
    agent : PokeAgent
        The PokeAgent holding the current model.
    """

    def __init__(
        self,
        opponents: List[Any],
        update_freq: int,
        agent: "PokeAgent",
        verbose: int = 0,
    ) -> None:
        """Store opponent list, update frequency, and agent reference."""
        super().__init__(verbose=verbose)
        self._opponents = opponents
        self._update_freq = update_freq
        self._agent = agent
        self._last_update = 0

    def _on_step(self) -> bool:
        """Check if it's time to refresh opponent weights and trigger update if so."""
        if self.num_timesteps - self._last_update >= self._update_freq:
            self._update_opponents()
            self._last_update = self.num_timesteps
        return True

    def _update_opponents(self) -> None:
        """Push current model weights to all frozen opponents."""
        for opp in self._opponents:
            # Only update if the opponent has an update_policy method
            if hasattr(opp, "update_policy") and self._agent.model is not None:
                opp.update_policy(self._agent.model)
            # else: opponent is a RandomPlayer – intentionally left random early on
        logger.info(
            f"[Step {self.num_timesteps}] Self-play opponents updated to latest policy."
        )
