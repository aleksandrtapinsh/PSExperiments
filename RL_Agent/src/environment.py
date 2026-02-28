"""
environment.py
==============
Pokemon Showdown RL environment for poke-env 0.11.x.

Architecture
------------
PokemonEnv(SinglesEnv)
    PettingZoo multi-agent env.  Both players (agent1, agent2) are internal
    server-connected bots.  Used with SingleAgentWrapper for SB3 training.

    Overrides:
      observation_space(agent) → 139-feature Box
      embed_battle(battle)     → float32 numpy array via observation.py
      calc_reward(battle)      → reward_computing_helper (incremental delta)

SelfPlayOpponent
    Lightweight class with choose_move() for use as the 'opponent' argument
    in SingleAgentWrapper.  Does NOT connect to the server — it only generates
    BattleOrder objects from the current PPO policy.  Starts as random and
    gets upgraded via update_policy() during training.

VsPlayerRunner(Player)
    Standalone server-connected player for human vs AI battles.
    Uses the PPO model for inference and performs online episodic learning
    after each battle.

Factory helpers
---------------
make_selfplay_env(cfg, acc1, acc2, opponent) -> SingleAgentWrapper (gym.Env)
make_random_opponent()                       -> SelfPlayOpponent (random policy)
build_server_config(cfg)                     -> ServerConfiguration
build_account_config(username, password)     -> AccountConfiguration
"""

from __future__ import annotations

import copy
import logging
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# poke-env 0.11.x imports
# ---------------------------------------------------------------------------
from sb3_contrib.common.wrappers import ActionMasker

from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env import AccountConfiguration, ServerConfiguration

try:
    from poke_env.battle.abstract_battle import AbstractBattle
except ImportError:
    from poke_env.environment.abstract_battle import AbstractBattle  # type: ignore

from .observation import embed_battle, OBS_SIZE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observation space bounds — all features live in [-1, 1] or [0, 1]
# ---------------------------------------------------------------------------
OBS_LOW = np.full(OBS_SIZE, -1.0, dtype=np.float32)
OBS_HIGH = np.full(OBS_SIZE, 1.0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------------------

def compute_action_mask(battle: Optional[AbstractBattle], n_actions: int) -> np.ndarray:
    """
    Return a boolean mask of shape (n_actions,) where True means the action
    is legal in the current battle state.

    The action space layout (poke-env 0.11 SinglesEnv) is:
      0-5   : switch to team slot 0-5
      6-9   : use move 0-3 (base)
      10-13 : use move 0-3 + mega
      14-17 : use move 0-3 + z-move
      18-21 : use move 0-3 + dynamax
      22-25 : use move 0-3 + terastallize

    The mask is computed by building the set of legal order strings from
    battle.available_moves / available_switches (+ modifiers), then checking
    each action's message against that set.  This exactly mirrors poke-env's
    own validity check, so the mask is always consistent.
    """
    if battle is None:
        return np.ones(n_actions, dtype=bool)

    # Build the set of valid order strings exactly as poke-env does internally.
    try:
        valid = {str(o) for o in battle.valid_orders}
    except Exception:
        return np.ones(n_actions, dtype=bool)

    if not valid:
        return np.ones(n_actions, dtype=bool)

    # Map each action index → its raw order string and check against valid set.
    # fake=True makes action_to_order skip its own validity check, returning the
    # raw (possibly illegal) order without logging a warning or falling back to a
    # random move — which is exactly what we need to build an accurate mask.
    mask = np.zeros(n_actions, dtype=bool)
    for action in range(n_actions):
        try:
            order = SinglesEnv.action_to_order(
                np.int64(action), battle, fake=True, strict=False
            )
            if str(order) in valid:
                mask[action] = True
        except Exception:
            pass

    # Safety: never return an all-False mask (e.g., at forced-switch end state).
    if not mask.any():
        mask[:] = True

    return mask


# ---------------------------------------------------------------------------
# Core RL environment (PettingZoo multi-agent, used via SingleAgentWrapper)
# ---------------------------------------------------------------------------

class PokemonEnv(SinglesEnv):
    """
    Pokemon Showdown RL environment for poke-env 0.11.x.

    Subclass SinglesEnv and implement the three required abstract methods:
      - observation_space(agent)
      - embed_battle(battle)
      - calc_reward(battle)
    """

    def __init__(self, reward_config: Dict[str, float], **kwargs: Any) -> None:
        """Initialise the env with reward weights and a shared observation space for both agents."""
        super().__init__(**kwargs)
        self._reward_cfg = reward_config
        # Single observation space shared by both agents
        _space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        # PettingZoo expects observation_spaces dict (used by SingleAgentWrapper)
        self.observation_spaces = {agent: _space for agent in self.possible_agents}

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        """Return the Box observation space for the given agent."""
        return self.observation_spaces[agent]

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """Convert the battle state to a 139-feature float32 observation vector."""
        return embed_battle(battle)

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Compute incremental delta reward using poke-env's reward_computing_helper."""
        return self.reward_computing_helper(
            battle,
            fainted_value=self._reward_cfg.get("fainted_value", 4.0),
            hp_value=self._reward_cfg.get("hp_value", 1.0),
            victory_value=self._reward_cfg.get("victory_value", 25.0),
            status_value=self._reward_cfg.get("status_value", 0.3),
        )


# ---------------------------------------------------------------------------
# Self-play opponent (used as 'opponent' in SingleAgentWrapper — NOT a server
# connection, just a choose_move() provider)
# ---------------------------------------------------------------------------

class SelfPlayOpponent:
    """
    Provides choose_move() for the second player slot in SingleAgentWrapper.

    Does not connect to the server.  Starts with a random policy and is
    upgraded to the latest PPO policy via update_policy().
    """

    def __init__(self) -> None:
        """Initialise with no policy (random play until update_policy() is called)."""
        self._policy: Optional[Any] = None  # SB3 PPO model or None

    def update_policy(self, model: Any) -> None:
        """Deep-copy the current PPO model for frozen-opponent inference."""
        self._policy = copy.deepcopy(model)
        logger.info("SelfPlayOpponent policy updated.")

    def choose_move(self, battle: AbstractBattle) -> DefaultBattleOrder:
        """Return a BattleOrder.  Falls back to random if no policy is set."""
        # When only /choose default is valid (e.g. last-Pokemon forced switch),
        # skip the policy entirely to avoid generating invalid-order warnings.
        try:
            if [str(o) for o in battle.valid_orders] == ["/choose default"]:
                return DefaultBattleOrder()
        except Exception:
            pass

        if self._policy is None:
            return self._random_move(battle)
        try:
            obs = embed_battle(battle)
            n_actions = self._policy.action_space.n
            masks = compute_action_mask(battle, n_actions)
            action, _ = self._policy.predict(
                obs[np.newaxis, :],
                action_masks=masks[np.newaxis, :],
                deterministic=False,
            )
            return SinglesEnv.action_to_order(
                np.int64(int(action[0])), battle, strict=False
            )
        except Exception as e:
            logger.debug(f"SelfPlayOpponent.choose_move fallback ({e})")
            return self._random_move(battle)

    @staticmethod
    def _random_move(battle: AbstractBattle) -> Any:
        """Choose a random legal move or switch."""
        opts = (battle.available_moves or []) + (battle.available_switches or [])
        if opts:
            return Player.create_order(random.choice(opts))
        return DefaultBattleOrder()


# ---------------------------------------------------------------------------
# Standalone vs-player runner with online episodic learning
# ---------------------------------------------------------------------------

class VsPlayerRunner(Player):
    """
    Server-connected Player that uses the PPO model for move selection and
    performs an offline episodic PPO update after each completed battle.

    Usage
    -----
    runner = VsPlayerRunner(model=ppo_model, reward_cfg=cfg["rewards"], ...)
    asyncio.run(runner.accept_challenges(opponent=None, n_challenges=100))
    """

    def __init__(
        self,
        model: Any,
        reward_cfg: Dict[str, float],
        update_every_n_battles: int = 1,
        n_update_epochs: int = 4,
        gamma: float = 0.99,
        log_dir: str = "logs",
        **kwargs: Any,
    ) -> None:
        """Initialise buffers, reward config, and TensorBoard writer for episodic learning."""
        super().__init__(**kwargs)
        self.model = model
        self._reward_cfg = reward_cfg
        self._update_every = update_every_n_battles
        self._n_epochs = n_update_epochs
        self._gamma = gamma

        # Per-battle trajectory storage
        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[int] = []
        self._logp_buf: List[float] = []
        self._val_buf: List[float] = []
        self._mask_buf: List[np.ndarray] = []
        self._reward_state: Dict[str, float] = {}  # battle_tag → last score

        self._battles_since_update = 0
        self._battle_count = 0
        self._recent_wins: List[int] = []  # 1=win, 0=loss for rolling win rate
        self._writer = SummaryWriter(log_dir=str(Path(log_dir) / "vsplayer"))

    # ------------------------------------------------------------------
    # Move selection with trajectory recording
    # ------------------------------------------------------------------

    def choose_move(self, battle: AbstractBattle):
        """Select a move using the PPO model and record the step in the trajectory buffer."""
        try:
            if [str(o) for o in battle.valid_orders] == ["/choose default"]:
                return DefaultBattleOrder()
        except Exception:
            pass

        obs = embed_battle(battle)
        masks = compute_action_mask(battle, self.model.action_space.n)
        obs_t = th.as_tensor(obs[np.newaxis, :], dtype=th.float32).to(self.model.device)

        with th.no_grad():
            actions_t, values_t, log_probs_t = self.model.policy.forward(
                obs_t, action_masks=masks[np.newaxis, :]
            )

        action = int(actions_t[0].item())
        self._obs_buf.append(obs)
        self._act_buf.append(action)
        self._logp_buf.append(float(log_probs_t[0].item()))
        self._val_buf.append(float(values_t[0].item()))
        self._mask_buf.append(masks)

        try:
            return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
        except Exception:
            opts = (battle.available_moves or []) + (battle.available_switches or [])
            if opts:
                return Player.create_order(random.choice(opts))
            return DefaultBattleOrder()

    # ------------------------------------------------------------------
    # Reward computation (delta per turn, mirrors reward_computing_helper)
    # ------------------------------------------------------------------

    def _compute_reward(self, battle: AbstractBattle) -> float:
        """Compute the delta reward for the current turn based on fainted/HP/victory signals."""
        tag = getattr(battle, "battle_tag", "default")
        my_fainted = sum(1 for p in battle.team.values() if p.fainted)
        opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
        my_hp = sum(p.current_hp_fraction for p in battle.team.values())
        opp_hp = sum(p.current_hp_fraction for p in battle.opponent_team.values())

        victory = 0.0
        if battle.finished:
            victory = (
                self._reward_cfg["victory_value"]
                if battle.won
                else -self._reward_cfg["victory_value"]
            )

        score = (
            self._reward_cfg["fainted_value"] * (opp_fainted - my_fainted)
            + self._reward_cfg["hp_value"] * (my_hp - opp_hp)
            + victory
        )
        prev = self._reward_state.get(tag, 0.0)
        self._reward_state[tag] = score
        return score - prev

    # ------------------------------------------------------------------
    # After-battle update
    # ------------------------------------------------------------------

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        """Trigger episodic PPO update, log outcome metrics, and clear trajectory buffers."""
        super()._battle_finished_callback(battle)

        # Compute final reward for this battle
        final_reward = self._compute_reward(battle)
        # Rebuild rewards list: all 0 for intermediate steps, final_reward at end
        n = len(self._obs_buf)
        rewards = [0.0] * (n - 1) + [final_reward] if n > 0 else []

        if n > 0 and len(rewards) == n:
            self._episodic_ppo_update(rewards)

        # Log result
        outcome = "WON" if battle.won else "LOST"
        logger.info(
            f"[VsPlayer] Battle finished — {outcome}. "
            f"Total: {self.n_won_battles}W/{self.n_lost_battles}L"
        )

        # TensorBoard logging
        self._battle_count += 1
        self._recent_wins.append(1 if battle.won else 0)
        if len(self._recent_wins) > 100:
            self._recent_wins.pop(0)
        rolling_win_rate = sum(self._recent_wins) / len(self._recent_wins)
        self._writer.add_scalar("vsplayer/win", float(battle.won), self._battle_count)
        self._writer.add_scalar("vsplayer/win_rate_100", rolling_win_rate, self._battle_count)
        self._writer.add_scalar("vsplayer/episode_return", final_reward, self._battle_count)
        self._writer.flush()

        # Reset trajectory
        self._obs_buf.clear()
        self._act_buf.clear()
        self._logp_buf.clear()
        self._val_buf.clear()
        self._mask_buf.clear()
        self._reward_state.pop(getattr(battle, "battle_tag", "default"), None)

        self._battles_since_update += 1
        if self._battles_since_update >= self._update_every:
            self._battles_since_update = 0

    # ------------------------------------------------------------------
    # Offline episodic PPO update
    # ------------------------------------------------------------------

    def _episodic_ppo_update(self, rewards: List[float]) -> None:
        """
        Perform a PPO-style policy gradient update on the collected episode.
        Uses Monte Carlo returns and the stored old log-probabilities for the
        importance-weight ratio clipping (PPO-clip objective).
        """
        if not self._obs_buf:
            return

        n = len(self._obs_buf)
        gamma = self._gamma
        device = self.model.device
        clip = self.model.clip_range(1.0)  # SB3 stores it as a Schedule

        # --- Compute discounted Monte Carlo returns ---
        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        obs_t = th.as_tensor(
            np.array(self._obs_buf), dtype=th.float32
        ).to(device)
        act_t = th.as_tensor(self._act_buf, dtype=th.long).to(device)
        ret_t = th.as_tensor(returns, dtype=th.float32).to(device)
        old_logp_t = th.as_tensor(self._logp_buf, dtype=th.float32).to(device)
        masks_arr = np.array(self._mask_buf)  # (n, n_actions)

        for _ in range(self._n_epochs):
            values, new_logp, entropy_t = self.model.policy.evaluate_actions(
                obs_t, act_t, action_masks=masks_arr
            )
            values = values.squeeze(-1)
            entropy = entropy_t.mean()

            # Advantage
            advantages = ret_t - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO clip ratio
            ratio = th.exp(new_logp - old_logp_t.detach())
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * th.clamp(ratio, 1.0 - clip, 1.0 + clip)
            pg_loss = th.max(pg_loss1, pg_loss2).mean()

            value_loss = ((ret_t - values) ** 2).mean()

            loss = (
                pg_loss
                + self.model.vf_coef * value_loss
                - self.model.ent_coef * entropy
            )

            self.model.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.model.policy.parameters(), self.model.max_grad_norm
            )
            self.model.policy.optimizer.step()

        logger.debug(
            f"[VsPlayer] Episodic update — {n} steps, "
            f"return={sum(returns):.2f}, loss={loss.item():.4f}"
        )
        self._writer.add_scalar("vsplayer/pg_loss", pg_loss.item(), self._battle_count)
        self._writer.add_scalar("vsplayer/value_loss", value_loss.item(), self._battle_count)
        self._writer.add_scalar("vsplayer/entropy", entropy.item(), self._battle_count)
        self._writer.add_scalar("vsplayer/total_loss", loss.item(), self._battle_count)
        self._writer.flush()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_server_config(cfg: Dict[str, Any]) -> ServerConfiguration:
    """Build a poke-env ServerConfiguration from the YAML config dict."""
    return ServerConfiguration(
        websocket_url=cfg["server"]["websocket_url"],
        authentication_url=cfg["server"]["auth_url"],
    )


def build_account_config(username: str, password: str = "") -> AccountConfiguration:
    """Build a poke-env AccountConfiguration, using None for empty passwords."""
    return AccountConfiguration(username, password if password else None)


def make_selfplay_env(
    cfg: Dict[str, Any],
    acc1: AccountConfiguration,
    acc2: AccountConfiguration,
    opponent: SelfPlayOpponent,
) -> SingleAgentWrapper:
    """
    Create a gymnasium-compatible env (via SingleAgentWrapper) for self-play.

    agent1 is the RL learner (actions come from SB3 PPO step()).
    agent2 is controlled by `opponent.choose_move()`.
    """
    env = PokemonEnv(
        reward_config=cfg["rewards"],
        account_configuration1=acc1,
        account_configuration2=acc2,
        server_configuration=build_server_config(cfg),
        battle_format=cfg["server"]["battle_format"],
        log_level=logging.WARNING,
        strict=False,
    )
    saw = SingleAgentWrapper(env, opponent)
    n_actions = saw.action_space.n

    def _mask_fn(e: SingleAgentWrapper) -> np.ndarray:
        # battle1 lives on the underlying PokeEnv, not on SingleAgentWrapper itself.
        return compute_action_mask(getattr(e.env, "battle1", None), n_actions)

    return ActionMasker(saw, _mask_fn)


def make_random_opponent() -> SelfPlayOpponent:
    """Return a SelfPlayOpponent that plays randomly (no model set)."""
    return SelfPlayOpponent()
