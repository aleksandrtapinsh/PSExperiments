"""
agent.py
========
Imitation Learning Pokémon Showdown player.

ILPlayer subclasses poke_env's Player and uses two trained Keras models
(move_model and switch_model) to select actions during live battles.

If the models are not found on disk, the agent falls back to random play
so it can still connect and accept battles even before training is done.

Model inputs
------------
Both models share the same signature:
    inputs  : [state_vector (OBS_SIZE,), action_mask (n_actions,)]
    outputs : softmax probability over n_actions

  move_model   — 4 outputs (move slots 0-3)
  switch_model — 5 outputs (switch slots 0-4)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

try:
    from poke_env.battle.abstract_battle import AbstractBattle
except ImportError:
    from poke_env.environment.abstract_battle import AbstractBattle  # type: ignore

from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder

from .observation import embed_battle_for_il

logger = logging.getLogger(__name__)


class ILPlayer(Player):
    """
    Server-connected player backed by trained Keras IL models.

    Falls back to random play if either model file is missing or TF is
    unavailable — useful for quick smoke-tests before training completes.
    """

    def __init__(
        self,
        move_model_path: str,
        switch_model_path: str,
        log_dir: str = "logs",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.move_model   = self._load_model(move_model_path)
        self.switch_model = self._load_model(switch_model_path)
        
        self._battle_count  = 0
        self._recent_wins: List[int] = []

        print(f"IN AGENT.PY {self.move_model}")
        print(f"IN AGENT.PY {self.switch_model}")

        # TensorBoard writer — optional, gracefully skipped if unavailable
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(Path(log_dir) / "il_agent"))
        except Exception:
            self._writer = None

        if self.move_model and self.switch_model:
            using = "move + switch Keras models"
        elif self.move_model:
            using = "move Keras model (switch model missing — force-switches will be random)"
        elif self.switch_model:
            using = "switch Keras model (move model missing — moves will be random)"
        else:
            using = "random play (no models loaded)"
        logger.info(f"ILPlayer initialised — {using}.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(path: str) -> Optional[Any]:
        if (path.endswith(".keras")):
            logger.info("Loading Neural Network _load_model")
            return ILPlayer._load_keras_model(path)
        elif (path.endswith(".pk1")):
            logger.info("Loading Random Forest _load_model")
            return ILPlayer._load_sklearn_model(path)
    
    @staticmethod
    def _load_keras_model(path: str) -> Optional[Any]:
        """Load a Keras model.  Returns None on any error."""
        try:
            import tensorflow as tf
        except ImportError:
            logger.warning("TensorFlow not installed — IL agent will play randomly.")
            return None

        p = Path(path)
        if not p.exists():
            logger.warning(f"Model not found at {p} — falling back to random play")
            return None
        try:
            model = tf.keras.models.load_model(str(p))
            logger.info(f"Loaded IL model: {p}")
            return model
        except Exception as e:
            logger.warning(f"Could not load model {p}: {e}")
            return None
    
    @staticmethod
    def _load_sklearn_model(path:str) -> Optional[Any]:
        try:
            import pickle
        except ImportError:
            print("Error importing pickle")
            return None
        
        p = Path(path)
        if not p.exists():
            logger.warning(f"Model not found at {p} — falling back to random play")
            return None
        try:
            with open(p, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded with pickle: {p}")
            return model
        except Exception as e:
            logger.warning(f"Could not load model {p}: {e}")
            return None

    # ------------------------------------------------------------------
    # Move selection
    # ------------------------------------------------------------------

    def choose_move(self, battle: AbstractBattle):
        """Select an action each turn using the IL model with random fallback."""
        # Forced default (e.g. last-Pokemon re-send, no real decision)
        try:
            if [str(o) for o in battle.valid_orders] == ["/choose default"]:
                return DefaultBattleOrder()
        except Exception:
            pass

        available_moves    = battle.available_moves or []
        available_switches = battle.available_switches or []

        # Force-switch: active Pokemon fainted, must send in a replacement
        force_switch = (not available_moves) and bool(available_switches)

        try:
            obs = embed_battle_for_il(battle)

            if force_switch:
                order = self._pick_switch(obs, available_switches)
                if order is not None:
                    return order
            else:
                # Default to using a move; fall through to switch if needed
                if available_moves and self.move_model is not None:
                    order = self._pick_move(obs, available_moves)
                    if order is not None:
                        return order
                if available_switches and self.switch_model is not None:
                    order = self._pick_switch(obs, available_switches)
                    if order is not None:
                        return order

        except Exception as e:
            logger.debug(f"IL inference error: {e}", exc_info=True)

        return self._random_move(battle)

    def _pick_move(self, obs: np.ndarray, available_moves: list) -> Optional[Any]:
        """Run the move model and return the best valid BattleOrder."""
        n = min(len(available_moves), 4)
        mask = np.zeros(4, dtype=np.float32)
        mask[:n] = 1.0
        try:
            probs = self.move_model.predict(
                [obs[np.newaxis], mask[np.newaxis]], verbose=0
            )[0]
            probs = probs * mask
            total = probs.sum()
            if total < 1e-9:
                return None
            best = int(np.argmax(probs))
            if best < n:
                return Player.create_order(available_moves[best])
        except Exception as e:
            logger.debug(f"Move model predict failed: {e}")
        return None

    def _pick_switch(self, obs: np.ndarray, available_switches: list) -> Optional[Any]:
        """Run the switch model and return the best valid BattleOrder."""
        n = min(len(available_switches), 5)
        mask = np.zeros(5, dtype=np.float32)
        mask[:n] = 1.0
        try:
            probs = self.switch_model.predict(
                [obs[np.newaxis], mask[np.newaxis]], verbose=0
            )[0]
            probs = probs * mask
            total = probs.sum()
            if total < 1e-9:
                return None
            best = int(np.argmax(probs))
            if best < n:
                return Player.create_order(available_switches[best])
        except Exception as e:
            logger.debug(f"Switch model predict failed: {e}")
        return None

    @staticmethod
    def _random_move(battle: AbstractBattle):
        """Random fallback: pick any legal move or switch."""
        opts = (battle.available_moves or []) + (battle.available_switches or [])
        if opts:
            return Player.create_order(random.choice(opts))
        return DefaultBattleOrder()

    # ------------------------------------------------------------------
    # Battle outcome logging
    # ------------------------------------------------------------------

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)

        outcome = "WON" if battle.won else "LOST"
        logger.info(
            f"[ILAgent] {outcome} — "
            f"{self.n_won_battles}W / {self.n_lost_battles}L"
        )

        self._battle_count += 1
        self._recent_wins.append(1 if battle.won else 0)
        if len(self._recent_wins) > 100:
            self._recent_wins.pop(0)

        if self._writer is not None:
            win_rate = sum(self._recent_wins) / max(len(self._recent_wins), 1)
            self._writer.add_scalar("il/win",          float(battle.won),  self._battle_count)
            self._writer.add_scalar("il/win_rate_100", win_rate,            self._battle_count)
            self._writer.flush()
