"""
observation.py
==============
Converts a live poke-env AbstractBattle into the IL feature vector.

The encoding matches training.py's encode_battle() exactly so that
the trained Keras models can be used for inference without retraining.

Feature vector layout (1357 floats):
  [0    - 111 ] My active pokemon      (POKEMON_DIM = 112)
  [112  - 671 ] My bench — 5 slots     (5 × 112 = 560)
  [672  - 783 ] Opponent active        (112)
  [784  - 1343] Opponent bench — 5 slots (560)
  [1344 - 1349] Weather one-hot        (6)
  [1350 - 1354] Terrain one-hot        (5)
  [1355]        Trick Room flag        (1)
  [1356]        Turn / 100, clipped    (1)

Total: 1357
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# poke-env imports — compatible with 0.9.x through 0.11.x
# ---------------------------------------------------------------------------
try:
    from poke_env.battle.abstract_battle import AbstractBattle
    from poke_env.battle.status import Status
    from poke_env.battle.weather import Weather
    from poke_env.battle.field import Field
except ImportError:
    from poke_env.environment.abstract_battle import AbstractBattle  # type: ignore
    from poke_env.environment.status import Status  # type: ignore
    from poke_env.environment.weather import Weather  # type: ignore
    from poke_env.environment.field import Field  # type: ignore

# ---------------------------------------------------------------------------
# Constants — must match training.py EXACTLY
# ---------------------------------------------------------------------------

STATUSES   = [None, "brn", "par", "slp", "frz", "psn", "tox"]
WEATHERS   = [None, "raindance", "sunnyday", "sandstorm", "hail", "snow"]
TERRAINS   = [None, "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]
TYPES      = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
              "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]
CATEGORIES = ["physical", "special", "status"]

MAX_MOVES   = 4
MAX_BENCH   = 5
MOVE_DIM    = 3 + len(TYPES) + len(CATEGORIES)                         # 24
POKEMON_DIM = 2 + len(STATUSES) + 7 + MAX_MOVES * MOVE_DIM             # 112
OBS_SIZE    = (2 * POKEMON_DIM) + (2 * MAX_BENCH * POKEMON_DIM) + len(WEATHERS) + len(TERRAINS) + 2  # 1357

# ---------------------------------------------------------------------------
# poke_env enum → training string mappings
# ---------------------------------------------------------------------------

_STATUS_STR = {
    Status.BRN: "brn",
    Status.PAR: "par",
    Status.PSN: "psn",
    Status.TOX: "tox",
    Status.SLP: "slp",
    Status.FRZ: "frz",
}

_WEATHER_STR: dict = {
    Weather.RAINDANCE: "raindance",
    Weather.SUNNYDAY:  "sunnyday",
    Weather.SANDSTORM: "sandstorm",
}
for _attr, _s in (("HAIL", "hail"), ("SNOW", "snow")):
    if hasattr(Weather, _attr):
        _WEATHER_STR[getattr(Weather, _attr)] = _s

_TERRAIN_STR = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN:   "grassyterrain",
    Field.MISTY_TERRAIN:    "mistyterrain",
    Field.PSYCHIC_TERRAIN:  "psychicterrain",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(val: Any, options: list) -> np.ndarray:
    vec = np.zeros(len(options), dtype=np.float32)
    try:
        vec[options.index(val)] = 1.0
    except ValueError:
        pass
    return vec


def _encode_move(move: Optional[Any]) -> np.ndarray:
    """24-dim feature vector for a single move slot."""
    if move is None:
        return np.zeros(MOVE_DIM, dtype=np.float32)

    # base_power / 250
    try:
        bp = float(move.base_power) / 250.0
    except (AttributeError, TypeError, ZeroDivisionError):
        bp = 0.0

    # accuracy (True = always-hit → 1.0)
    try:
        acc = move.accuracy
        acc = 1.0 if acc is True else float(acc)
    except (AttributeError, TypeError):
        acc = 1.0

    # PP fraction
    try:
        max_pp = max(int(move.max_pp or 1), 1)
        cur_pp = getattr(move, "current_pp", max_pp)
        pp_frac = float(cur_pp) / float(max_pp)
    except (AttributeError, TypeError, ZeroDivisionError):
        pp_frac = 1.0

    scalar = np.array([bp, acc, pp_frac], dtype=np.float32)

    # Type one-hot
    try:
        type_str = move.type.name.capitalize() if move.type is not None else None
    except AttributeError:
        type_str = None
    type_vec = _one_hot(type_str, TYPES)

    # Category one-hot
    try:
        cat_str = move.category.name.lower() if move.category is not None else None
    except AttributeError:
        cat_str = None
    cat_vec = _one_hot(cat_str, CATEGORIES)

    return np.concatenate([scalar, type_vec, cat_vec])


def _encode_pokemon(pokemon: Optional[Any]) -> np.ndarray:
    """112-dim feature vector for a single Pokemon."""
    if pokemon is None:
        return np.zeros(POKEMON_DIM, dtype=np.float32)

    hp_frac = float(getattr(pokemon, "current_hp_fraction", 0.0))
    fainted  = float(bool(getattr(pokemon, "fainted", False)))
    scalar   = np.array([hp_frac, fainted], dtype=np.float32)

    status_str = _STATUS_STR.get(getattr(pokemon, "status", None), None)
    status_vec = _one_hot(status_str, STATUSES)

    boosts = getattr(pokemon, "boosts", {}) or {}
    boost_vec = np.array([
        boosts.get("atk",      0) / 6.0,
        boosts.get("def",      0) / 6.0,
        boosts.get("spa",      0) / 6.0,
        boosts.get("spd",      0) / 6.0,
        boosts.get("spe",      0) / 6.0,
        boosts.get("accuracy", 0) / 6.0,
        boosts.get("evasion",  0) / 6.0,
    ], dtype=np.float32)

    moves = list((getattr(pokemon, "moves", {}) or {}).values())[:MAX_MOVES]
    move_vecs = [_encode_move(m) for m in moves]
    while len(move_vecs) < MAX_MOVES:
        move_vecs.append(np.zeros(MOVE_DIM, dtype=np.float32))

    return np.concatenate([scalar, status_vec, boost_vec, *move_vecs])


def _encode_bench(bench: list) -> np.ndarray:
    """Encode up to MAX_BENCH bench pokemon; zero-pad remaining slots."""
    vecs = [_encode_pokemon(p) for p in bench[:MAX_BENCH]]
    while len(vecs) < MAX_BENCH:
        vecs.append(np.zeros(POKEMON_DIM, dtype=np.float32))
    return np.concatenate(vecs)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def embed_battle_for_il(battle: AbstractBattle) -> np.ndarray:
    """
    Convert a live poke-env battle to a float32 array of shape (OBS_SIZE,).
    Encoding matches training.py's encode_battle() so trained models work directly.
    """
    my_active  = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    my_bench = [
        p for p in battle.team.values()
        if p is not my_active and not getattr(p, "fainted", False)
    ]
    opp_bench = [
        p for p in battle.opponent_team.values()
        if p is not opp_active and not getattr(p, "fainted", False)
    ]

    # Weather
    raw_weather = getattr(battle, "weather", None)
    if isinstance(raw_weather, dict):
        raw_weather = next(iter(raw_weather), None)
    weather_str = _WEATHER_STR.get(raw_weather, None)

    # Terrain and Trick Room from battle.fields dict
    fields = getattr(battle, "fields", {}) or {}
    terrain_str = None
    for field_enum, t_str in _TERRAIN_STR.items():
        if field_enum in fields:
            terrain_str = t_str
            break
    trick_room = float(Field.TRICK_ROOM in fields)

    turn_norm = float(min(getattr(battle, "turn", 0), 100)) / 100.0

    obs = np.concatenate([
        _encode_pokemon(my_active),
        _encode_bench(my_bench),
        _encode_pokemon(opp_active),
        _encode_bench(opp_bench),
        _one_hot(weather_str, WEATHERS),
        _one_hot(terrain_str, TERRAINS),
        [trick_room],
        [turn_norm],
    ]).astype(np.float32)

    assert len(obs) == OBS_SIZE, f"IL obs size mismatch: got {len(obs)}, expected {OBS_SIZE}"
    return obs
