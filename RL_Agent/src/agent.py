"""
agent.py
========
PPO-based Pokémon Showdown RL agent for poke-env 0.11.x.

Two training modes
------------------
selfplay
    Two SingleAgentWrapper envs run parallel battles (each agent1 vs a frozen
    SelfPlayOpponent).  Both feed experience to the same PPO rollout buffer for
    ~2x data per wall-clock second.  The frozen opponent is updated every
    `selfplay_opponent_update_freq` steps.

vs_player
    A VsPlayerRunner (standalone Player) connects to the server and accepts
    challenges from any human player.  The PPO model is used for inference and
    updated after each battle via offline episodic PPO.

Persistence
-----------
  models/poke_ppo.zip  — saved on exit and after checkpoints; loaded on startup.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as EvalCallback
except ImportError:
    from stable_baselines3.common.callbacks import EvalCallback  # type: ignore

from poke_env import AccountConfiguration
from poke_env.player import POKE_LOOP

from .environment import (
    SelfPlayOpponent,
    VsPlayerRunner,
    build_account_config,
    build_server_config,
    make_selfplay_env,
    make_random_opponent,
    cleanup_vec_env,
    cleanup_runner,
)
from .observation import OBS_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network policy configuration
# ---------------------------------------------------------------------------

def _policy_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build SB3 policy_kwargs (net architecture and activation function) from config."""
    act_fn_map = {
        "tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU,
    }
    return dict(
        net_arch=dict(
            pi=cfg["model"]["net_arch_pi"],
            vf=cfg["model"]["net_arch_vf"],
        ),
        activation_fn=act_fn_map.get(cfg["model"]["activation_fn"].lower(), nn.Tanh),
    )


# ---------------------------------------------------------------------------
# PokeAgent
# ---------------------------------------------------------------------------

class PokeAgent:
    """
    Manages the PPO model lifecycle: build, load, save, selfplay training,
    and vs-player online learning.
    """

    def __init__(self, cfg: Dict[str, Any], device: str = "auto") -> None:
        """Store config and device; model is built lazily via load() or train methods."""
        self.cfg = cfg
        self.device = device
        self.model: Optional[MaskablePPO] = None
        self._save_path: str = cfg["model"]["save_path"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, env: VecEnv) -> MaskablePPO:
        """Instantiate a fresh MaskablePPO model from config hyperparameters."""
        t = self.cfg["training"]
        return MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=t["learning_rate"],
            n_steps=t["n_steps"],
            batch_size=t["batch_size"],
            n_epochs=t["n_epochs"],
            gamma=t["gamma"],
            gae_lambda=t["gae_lambda"],
            clip_range=t["clip_range"],
            ent_coef=t["ent_coef"],
            vf_coef=t["vf_coef"],
            max_grad_norm=t["max_grad_norm"],
            policy_kwargs=_policy_kwargs(self.cfg),
            verbose=self.cfg["logging"]["verbose"],
            tensorboard_log=self.cfg["logging"]["log_dir"],
            device=self.device,
        )

    def save(self, path: Optional[str] = None) -> None:
        """Save the current PPO model to disk."""
        if self.model is None:
            logger.warning("No model to save.")
            return
        p = path or self._save_path
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(p)
        logger.info(f"Model saved → {p}.zip")

    def load(self, env: VecEnv, path: Optional[str] = None) -> bool:
        """Load from disk if available; otherwise initialise fresh.  Returns True if loaded."""
        p = path or self._save_path
        zip_path = p if p.endswith(".zip") else p + ".zip"
        if os.path.exists(zip_path):
            logger.info(f"Loading model from {zip_path}")
            try:
                self.model = MaskablePPO.load(
                    p, env=env, device=self.device,
                    tensorboard_log=self.cfg["logging"]["log_dir"],
                )
                return True
            except ValueError as e:
                logger.warning(
                    f"Could not load existing model — policy class mismatch ({e}). "
                    "The old checkpoint was trained with plain PPO and is incompatible "
                    "with MaskablePPO. Starting from scratch."
                )
        logger.info("No saved model — starting from scratch.")
        self.model = self._build_model(env)
        return False

    # ------------------------------------------------------------------
    # Self-play training
    # ------------------------------------------------------------------

    def train_selfplay(self) -> None:
        """
        Train via self-play: two parallel SingleAgentWrapper envs, both against
        a SelfPlayOpponent that starts random and gets updated to the current
        policy every `selfplay_opponent_update_freq` steps.
        """
        cfg = self.cfg
        t_cfg = cfg["training"]
        l_cfg = cfg["logging"]

        # Two frozen opponents (start random, promoted during training)
        opp_A = make_random_opponent()
        opp_B = make_random_opponent()

        acc_A = build_account_config(cfg["server"]["agent1_username"])
        acc_B = build_account_config(cfg["server"]["agent2_username"])
        # Two separate accounts are also needed for the internal agent2 slots
        acc_A2 = build_account_config(cfg["server"]["internal_opp_a_username"])
        acc_B2 = build_account_config(cfg["server"]["internal_opp_b_username"])

        def _make_A():
            return Monitor(make_selfplay_env(cfg, acc_A, acc_A2, opp_A))

        def _make_B():
            return Monitor(make_selfplay_env(cfg, acc_B, acc_B2, opp_B))

        vec_env = DummyVecEnv([_make_A, _make_B])

        loaded = self.load(vec_env)

        # Eval env: agent vs random opponent
        eval_opp = make_random_opponent()
        eval_env = Monitor(
            make_selfplay_env(
                cfg,
                build_account_config(cfg["server"]["eval1_username"]),
                build_account_config(cfg["server"]["eval2_username"]),
                eval_opp,
            )
        )
        #A
        eval_vec = DummyVecEnv([lambda: eval_env])

        # Callbacks
        from .utils import BattleMetricsCallback, SelfPlayOpponentUpdateCallback

        ckpt_cb = CheckpointCallback(
            save_freq=l_cfg["checkpoint_freq"] // t_cfg["n_envs_selfplay"],
            save_path=str(Path(self._save_path).parent),
            name_prefix="poke_ppo_ckpt",
        )
        eval_cb = EvalCallback(
            eval_vec,
            best_model_save_path=str(Path(self._save_path).parent / "best"),
            log_path=l_cfg["log_dir"],
            eval_freq=l_cfg["eval_freq"] // t_cfg["n_envs_selfplay"],
            n_eval_episodes=l_cfg["n_eval_episodes"],
            deterministic=True,
            render=False,
        )
        metrics_cb = BattleMetricsCallback(
            log_dir=l_cfg["log_dir"],
            battle_log_freq=l_cfg.get("battle_log_freq", 10),
        )
        sp_cb = SelfPlayOpponentUpdateCallback(
            opponents=[opp_A, opp_B],
            update_freq=t_cfg["selfplay_opponent_update_freq"],
            agent=self,
        )

        try:
            self.model.learn(
                total_timesteps=t_cfg["total_timesteps"],
                callback=CallbackList([ckpt_cb, eval_cb, metrics_cb, sp_cb]),
                reset_num_timesteps=not loaded,
                tb_log_name="selfplay",
                progress_bar=True,
            )
        finally:
            self.save()
            # Forfeit any in-progress battles before disconnecting so accounts
            # are freed immediately instead of waiting for the server timeout.
            cleanup_vec_env(vec_env)
            cleanup_vec_env(eval_vec)
            logger.info("Self-play training finished.")

    # ------------------------------------------------------------------
    # vs-player training
    # ------------------------------------------------------------------

    def train_vs_player(self) -> None:
        """
        Connect as a standalone Player, accept challenges from any human, and
        update the PPO model online after each battle (episodic PPO).

        The model is loaded from disk first so existing knowledge is preserved.
        A dummy VecEnv (single random-opponent battle) is used just to
        satisfy SB3's model initialisation; the actual learning happens via
        VsPlayerRunner._episodic_ppo_update().
        """
        cfg = self.cfg
        t_cfg = cfg["training"]
        l_cfg = cfg["logging"]

        # Build a minimal env to initialise / load the SB3 model.
        # Use eval accounts so agent1_username is free when VsPlayerRunner connects.
        dummy_opp = make_random_opponent()
        dummy_env = Monitor(
            make_selfplay_env(
                cfg,
                build_account_config(cfg["server"]["eval1_username"]),
                build_account_config(cfg["server"]["eval2_username"]),
                dummy_opp,
            )
        )
        vec_env = DummyVecEnv([lambda: dummy_env])
        self.load(vec_env)
        # We only used vec_env to load the model — close the dummy env now
        vec_env.close()

        # Build the standalone VsPlayerRunner
        runner = VsPlayerRunner(
            model=self.model,
            reward_cfg=cfg["rewards"],
            update_every_n_battles=1,
            n_update_epochs=self.cfg["training"]["n_epochs"],
            gamma=self.cfg["training"]["gamma"],
            log_dir=l_cfg["log_dir"],
            account_configuration=build_account_config(cfg["server"]["agent1_username"]),
            server_configuration=build_server_config(cfg),
            battle_format=cfg["server"]["battle_format"],
            log_level=logging.WARNING,
        )

        logger.info(
            f"VsPlayer runner started as '{cfg['server']['agent1_username']}'. "
            "Waiting for challenges..."
        )

        from .utils import BattleMetricsCallback
        battles_played = 0
        checkpoint_freq = l_cfg["checkpoint_freq"]

        async def _accept_loop():
            nonlocal battles_played
            while battles_played < t_cfg["total_vs_battles"]:
                # Accept one challenge at a time; loop for many battles
                await runner.accept_challenges(
                    opponent=None, n_challenges=1
                )
                battles_played += 1
                if battles_played % max(1, checkpoint_freq // 20) == 0:
                    self.save()
                    logger.info(
                        f"[VsPlayer] {battles_played} battles played — model saved. "
                        f"Wins: {runner.n_won_battles}, "
                        f"Losses: {runner.n_lost_battles}"
                    )

        try:
            asyncio.get_event_loop().run_until_complete(_accept_loop())
        except KeyboardInterrupt:
            logger.info("VsPlayer session interrupted by user.")
        finally:
            self.save()
            # Forfeit any battle still in progress before disconnecting.
            cleanup_runner(runner)
            logger.info("vs-player session finished.")
    
    # ------------------------------------------------------------------
    # vs-IL training
    # ------------------------------------------------------------------
    def train_vs_il(self, il_username: Optional[str] = None) -> None:
        """
        Actively challenge the IL bot on the server and learn from each battle. 
        The RL agent sends challenges instead of waiting for them; otherwise 
        the learning logic is identical to train_vs_player().

        Prerequisites
        - The IL bot must already be logged in and running on the same server.
        - il_agent_username in config.yaml must match the IL bot's username.

        Usage
        -----
          python main.py vs_il --timesteps num_of_steps
        """
        cfg = self.cfg
        t_cfg = cfg["training"]
        l_cfg = cfg["logging"]

        il_username = il_username or cfg["server"].get("il_agent_username")
        if not il_username:
            raise ValueError(
                "il_agent_username must be set in config.yaml under 'server:', "
                "or passed directly as an argument."
            )

        # Use eval accounts for the dummy env so agent2_username stays free
        # for the actual VsPlayerRunner connection (same pattern as train_vs_player)
        dummy_opp = make_random_opponent()
        dummy_env = Monitor(
            make_selfplay_env(
                cfg,
                build_account_config(cfg["server"]["eval1_username"]),
                build_account_config(cfg["server"]["eval2_username"]),
                dummy_opp,
            )
        )
        vec_env = DummyVecEnv([lambda: dummy_env])
        self.load(vec_env)
        vec_env.close()

        
        #VsPlayerRunner handles move selection + episodic PPO update after each battle.
        #the only difference is we call send_challenges() instead of
        #accept_challenges(), so the RL agent is the one initiating each battle.
        
        runner = VsPlayerRunner(
            model=self.model,
            reward_cfg=cfg["rewards"],
            update_every_n_battles=1,
            n_update_epochs=cfg["training"]["n_epochs"],
            gamma=cfg["training"]["gamma"],
            log_dir=l_cfg["log_dir"],
            account_configuration=build_account_config(cfg["server"]["agent2_username"]),
            server_configuration=build_server_config(cfg),
            battle_format=cfg["server"]["battle_format"],
            log_level=logging.WARNING,
        )

        logger.info(
            f"VS-IL mode started. "
            f"RL agent '{cfg['server']['agent2_username']}' will challenge "
            f"IL bot '{il_username}' on {cfg['server']['websocket_url']}."
        )

        battles_played = 0
        checkpoint_freq = l_cfg["checkpoint_freq"]

        async def _challenge_loop():
            nonlocal battles_played
            while battles_played < t_cfg["total_vs_bettles"]:
                
                #Actively send a challenge to the IL bot and wait for it to accept.
                await runner.send_challenges(
                    opponent=il_username,
                    n_challenges=1,
                )
                battles_played += 1
                if battles_played % max(1, checkpoint_freq // 20) == 0:
                    self.save()
                    logger.info(
                        f"[VS-IL] {battles_played} battles played — model saved. "
                        f"Wins: {runner.n_won_battles}, "
                        f"Losses: {runner.n_lost_battles}"
                    )

        try:
            asyncio.get_event_loop().run_until_complete(_challenge_loop())
        except KeyboardInterrupt:
            logger.info("VS-IL session interrupted by user.")
        finally:
            self.save()
            cleanup_runner(runner)
            logger.info("VS-IL training finished.")