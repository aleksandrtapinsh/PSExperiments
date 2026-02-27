"""
observation.py
==============
Converts a poke-env AbstractBattle into a fixed-length numpy feature vector.

Compatible with poke-env 0.11.x.

Feature vector layout (139 floats, all in [-1, 1] or [0, 1]):
  [  0      ] My active pokemon HP fraction
  [  1 -  18] My active pokemon type multi-hot (18 types)
  [ 19 -  25] My active pokemon status one-hot (6 + no-status)
  [ 26 -  32] My active pokemon stat boosts / 6 → [-1, 1]
  [ 33 -  60] My 4 available moves x 7 features each
  [ 61      ] Opponent active HP fraction
  [ 62 -  79] Opponent types
  [ 80 -  86] Opponent status
  [ 87 -  93] Opponent stat boosts
  [ 94 -  99] My team HP fractions (6 slots)
  [100 - 105] My team fainted flags
  [106 - 111] My team available-switch flags
  [112 - 117] Opponent team HP fractions (0 if unrevealed)
  [118 - 123] Opponent team fainted flags
  [124 - 131] Weather one-hot (8 slots)
  [132 - 136] Terrain one-hot (5 slots)
  [137      ] Can Terastallize flag
  [138      ] Turn count normalised (turn / 50, clipped to [0, 1])

Total: 139 features
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# poke-env 0.11.x imports
# ---------------------------------------------------------------------------
try:
    from poke_env.battle.abstract_battle import AbstractBattle
except ImportError:
    from poke_env.environment.abstract_battle import AbstractBattle  # type: ignore

try:
    from poke_env.data.gen_data import GenData
    GEN9_DATA = GenData.from_gen(9)
    TYPE_CHART = GEN9_DATA.type_chart
except Exception:
    GEN9_DATA = None
    TYPE_CHART = None

# PokemonType, Status, Weather, Field — try multiple import paths
try:
    from poke_env.battle.pokemon_type import PokemonType
except ImportError:
    from poke_env.environment.pokemon_type import PokemonType  # type: ignore

try:
    from poke_env.battle.status import Status
except ImportError:
    from poke_env.environment.status import Status  # type: ignore

try:
    from poke_env.battle.weather import Weather
except ImportError:
    from poke_env.environment.weather import Weather  # type: ignore

try:
    from poke_env.battle.field import Field
except ImportError:
    from poke_env.environment.field import Field  # type: ignore

try:
    from poke_env.battle.move_category import MoveCategory
except ImportError:
    from poke_env.environment.move_category import MoveCategory  # type: ignore

# ---------------------------------------------------------------------------
# Ordered enumerations (order must remain stable across calls)
# ---------------------------------------------------------------------------

POKEMON_TYPES: List[PokemonType] = [
    PokemonType.NORMAL, PokemonType.FIRE, PokemonType.WATER, PokemonType.ELECTRIC,
    PokemonType.GRASS, PokemonType.ICE, PokemonType.FIGHTING, PokemonType.POISON,
    PokemonType.GROUND, PokemonType.FLYING, PokemonType.PSYCHIC, PokemonType.BUG,
    PokemonType.ROCK, PokemonType.GHOST, PokemonType.DRAGON, PokemonType.DARK,
    PokemonType.STEEL, PokemonType.FAIRY,
]  # 18 types

STATUSES: List[Optional[Status]] = [
    Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX,
    None,  # no status → index 6
]  # 7 slots

BOOSTS_ORDER: List[str] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


def _build_weather_list() -> List[Optional[Weather]]:
    """Build weather list; handles gen9 SNOW vs gen8 HAIL rename."""
    weathers: List[Optional[Weather]] = [
        None,
        Weather.DESOLATELAND, Weather.PRIMORDIALSEA, Weather.DELTASTREAM,
        Weather.RAINDANCE, Weather.SANDSTORM, Weather.SUNNYDAY,
    ]
    for attr in ("SNOW", "HAIL"):  # gen9 renamed HAIL → SNOW
        if hasattr(Weather, attr):
            val = getattr(Weather, attr)
            if val not in weathers:
                weathers.append(val)
            break
    return weathers


WEATHERS: List[Optional[Weather]] = _build_weather_list()
_WEATHER_SLOTS = 8  # always allocate 8 slots regardless of version

TERRAINS: List[Optional[Field]] = [
    None,
    Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN,
]  # 5 slots

# Feature vector total size — must match the layout above
OBS_SIZE: int = 139

# ---------------------------------------------------------------------------
# Helper encoders
# ---------------------------------------------------------------------------

def _encode_types(mon) -> np.ndarray:
    """18-element multi-hot type vector."""
    vec = np.zeros(18, dtype=np.float32)
    if mon is None:
        return vec
    for i, t in enumerate(POKEMON_TYPES):
        if mon.type_1 == t or mon.type_2 == t:
            vec[i] = 1.0
    return vec


def _encode_status(mon) -> np.ndarray:
    """7-element one-hot status vector (index 6 = no status)."""
    vec = np.zeros(7, dtype=np.float32)
    if mon is None:
        vec[6] = 1.0
        return vec
    s = getattr(mon, "status", None)
    try:
        idx = STATUSES.index(s)
    except ValueError:
        idx = 6
    vec[idx] = 1.0
    return vec


def _encode_boosts(mon) -> np.ndarray:
    """7-element boost vector, each in [-1, 1]."""
    vec = np.zeros(7, dtype=np.float32)
    if mon is None:
        return vec
    boosts = getattr(mon, "boosts", {})
    for i, stat in enumerate(BOOSTS_ORDER):
        vec[i] = boosts.get(stat, 0) / 6.0
    return vec


def _type_effectiveness(move, opponent) -> float:
    """Type effectiveness as log2 score clipped to [-1, 1]."""
    if opponent is None or move.type is None:
        return 0.0
    try:
        if TYPE_CHART is not None:
            mult = move.type.damage_multiplier(
                opponent.type_1, opponent.type_2, type_chart=TYPE_CHART
            )
        else:
            mult = move.type.damage_multiplier(opponent.type_1, opponent.type_2)
        return float(np.clip(math.log2(mult) / 2.0, -1.0, 1.0)) if mult > 0 else -1.0
    except Exception:
        return 0.0


def _encode_move(move, opponent) -> np.ndarray:
    """
    7-element feature vector for one move slot.
      [0] base_power / 300
      [1] type effectiveness (log2 scale, [-1, 1])
      [2] is physical
      [3] is special
      [4] accuracy (1.0 for always-hit)
      [5] priority / 5
      [6] PP fraction
    """
    if move is None:
        return np.zeros(7, dtype=np.float32)
    vec = np.zeros(7, dtype=np.float32)
    try:
        vec[0] = min(move.base_power / 300.0, 1.0)
    except (KeyError, AttributeError):
        pass
    vec[1] = _type_effectiveness(move, opponent)
    try:
        cat = move.category
        vec[2] = 1.0 if cat == MoveCategory.PHYSICAL else 0.0
        vec[3] = 1.0 if cat == MoveCategory.SPECIAL else 0.0
    except (KeyError, AttributeError):
        pass
    try:
        acc = move.accuracy
        vec[4] = 1.0 if acc is True else float(acc)
    except (KeyError, AttributeError):
        vec[4] = 1.0
    try:
        priority = move.priority
    except (KeyError, AttributeError):
        priority = 0
    vec[5] = float(np.clip(priority / 5.0, -1.0, 1.0))
    try:
        pp = move.pp or 1
        vec[6] = float(getattr(move, "current_pp", pp)) / float(pp)
    except (KeyError, AttributeError):
        vec[6] = 1.0
    return vec


def _encode_weather(battle) -> np.ndarray:
    """8-element one-hot weather vector."""
    vec = np.zeros(_WEATHER_SLOTS, dtype=np.float32)
    w = getattr(battle, "weather", None)
    if isinstance(w, dict):
        w = next(iter(w), None) if w else None
    try:
        idx = WEATHERS.index(w)
        if idx < _WEATHER_SLOTS:
            vec[idx] = 1.0
    except ValueError:
        vec[0] = 1.0
    return vec


def _encode_terrain(battle) -> np.ndarray:
    """5-element one-hot terrain vector."""
    vec = np.zeros(5, dtype=np.float32)
    fields = getattr(battle, "fields", {})
    terrain = None
    for t in TERRAINS[1:]:
        if t in fields:
            terrain = t
            break
    try:
        vec[TERRAINS.index(terrain)] = 1.0
    except ValueError:
        vec[0] = 1.0
    return vec

# ---------------------------------------------------------------------------
# Main embedding function
# ---------------------------------------------------------------------------

def embed_battle(battle: AbstractBattle) -> np.ndarray:
    """
    Convert a poke-env battle into a fixed-size float32 numpy array of shape (OBS_SIZE,).
    All values are clipped to [-1, 1] or [0, 1].  Unknown slots are zero-padded.
    """
    f: List[float] = []

    # 1. My active pokemon — 33 features (1+18+7+7)
    me = battle.active_pokemon
    f.append(me.current_hp_fraction if me is not None else 0.0)
    f.extend(_encode_types(me))
    f.extend(_encode_status(me))
    f.extend(_encode_boosts(me))

    # 2. My available moves — 28 features (4 × 7)
    opp = battle.opponent_active_pokemon
    moves = battle.available_moves or []
    for i in range(4):
        f.extend(_encode_move(moves[i] if i < len(moves) else None, opp))

    # 3. Opponent active pokemon — 33 features
    f.append(opp.current_hp_fraction if opp is not None else 0.0)
    f.extend(_encode_types(opp))
    f.extend(_encode_status(opp))
    f.extend(_encode_boosts(opp))

    # 4. My team — 18 features (6 × 3: HP, fainted, can_switch)
    my_team = list(battle.team.values())
    switches = set(battle.available_switches or [])
    for i in range(6):
        if i < len(my_team):
            mon = my_team[i]
            f.append(mon.current_hp_fraction)
            f.append(1.0 if mon.fainted else 0.0)
            f.append(1.0 if mon in switches else 0.0)
        else:
            f.extend([0.0, 1.0, 0.0])

    # 5. Opponent team — 12 features (6 × 2: HP, fainted)
    opp_team = list(battle.opponent_team.values())
    for i in range(6):
        if i < len(opp_team):
            mon = opp_team[i]
            f.append(mon.current_hp_fraction)
            f.append(1.0 if mon.fainted else 0.0)
        else:
            f.extend([0.0, 0.0])

    # 6. Weather — 8 features
    f.extend(_encode_weather(battle))

    # 7. Terrain — 5 features
    f.extend(_encode_terrain(battle))

    # 8. Can Terastallize — 1 feature
    f.append(1.0 if getattr(battle, "can_tera", False) else 0.0)

    # 9. Turn count normalised — 1 feature
    f.append(float(np.clip(getattr(battle, "turn", 0) / 50.0, 0.0, 1.0)))

    obs = np.array(f, dtype=np.float32)
    assert len(obs) == OBS_SIZE, f"Observation size mismatch: {len(obs)} != {OBS_SIZE}"
    return obs
