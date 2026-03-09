"""
parser.py

Parses replay to turn level (state, action) records for imitation learning

Expected log format : Showdown protocol text logs (|tag|args...)
Usage:
    python parser.py --logs ../../logs --out ./dataset.jsonl
    python parser.py --logs ../../logs --out ./dataset.jsonl --format json
"""

import json
import re
import argparse
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

#Data Structures

@dataclass
class MoveFeatures:
    name: str
    base_power: int = 0
    accuracy: float = 1.0
    pp: int = 0
    current_pp: int = 0
    type_: str = "normal"
    category: str = "physical"
    priority: int = 0
    makes_contact: bool = False
    
    def to_dict(self):
        d = asdict(self)
        d["type"] = d.pop["type_"]
        return d
#defaults to normal type physical move with 0 base power

@dataclass
class PokemonState:
    species: str = ""
    hp_frac: float = 1.0
    fainted: bool = False
    status: Optional[str] = None # "brn", "par", "psn", "tox", "slp", "frz", None
    boosts: dict = field(default_factory=lambda:{
        "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0
    })
    moves: list = field(default_factory=list) #list of MoveFeature dicts
    ability: Optional[str] = None
    item: Optional[str] = None
    type1: Optional[str] = None
    type2: Optional[str] = None
    level: int = 100
    is_active: bool = False

@dataclass
class SideState:
    active: Optional[PokemonState] = None
    bench: list = field(default_factory=list) # list of PokemonState dicts
    conditions: list = field(default_factory=list) # stealthrocks / spikes

@dataclass 
class FieldState:
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: bool = False
    gravity: bool = False
    turn: int = 0

@dataclass
class TurnRecord:
    """One (state, action) training example """
    battle_id: str
    turn_number: int
    player: str         #p1 or p2
    perspective: str    #whose POV we're recording

    #state visibile to acting player at decision time
    my_active: dict = field(default_factory=dict)
    my_bench: list = field(default_factory=list)
    my_conditions: list = field(default_factory=list)

    opp_active: dict = field(default_factory=dict)
    opp_bench: list = field(default_factory=list)
    opp_conditions: list = field(default_factory=list)

    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: bool = False
    turn: int = 0

    # Available actions at this decision point
    available_moves: list = field(default_factory=list)
    available_switches: list = field(default_factory=list)
    force_switch: bool = False     # true when a pokemon fainted

    # Ground truth action from the replay
    action_type: str = ""          # "move" or "switch"
    action_name: str = ""          # move name or species switching in
    action_slot: int = 0           # 1-indexed slot in available list

    # Battle outcome (filled in after parsing is complete)
    winner: Optional[str] = None

# ---------------------------------------------------------------------------
# Move database — backed by PokeAPI with local disk cache
# ---------------------------------------------------------------------------

POKEAPI_BASE = "https://pokeapi.co/api/v2"
DEFAULT_CACHE_PATH = Path("move_cache.json")

# In-memory cache: move slug -> feature dict
_MOVE_CACHE: dict = {}
_CACHE_PATH: Path = DEFAULT_CACHE_PATH


def set_cache_path(path: Path):
    """Call this before parsing if you want a non-default cache location."""
    global _CACHE_PATH
    _CACHE_PATH = path


def _load_disk_cache():
    """Load previously fetched moves from disk into memory."""
    global _MOVE_CACHE
    if _CACHE_PATH.exists():
        try:
            _MOVE_CACHE = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            print(f"[MoveDB] Loaded {len(_MOVE_CACHE)} moves from cache ({_CACHE_PATH})")
        except Exception as e:
            print(f"[MoveDB] Could not read cache file: {e}")
            _MOVE_CACHE = {}


def _save_disk_cache():
    """Flush in-memory cache to disk."""
    try:
        _CACHE_PATH.write_text(
            json.dumps(_MOVE_CACHE, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"[MoveDB] Warning: could not save cache: {e}")


def _fetch_move_from_api(slug: str) -> dict:
    """
    Fetch a single move from PokeAPI and return a normalised feature dict.
    Raises urllib.error.URLError on network failure.
    """
    url = f"{POKEAPI_BASE}/move/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "pokemon-il-parser/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # PokeAPI accuracy is 0-100 int or null (always-hit moves)
    raw_acc = data.get("accuracy")
    accuracy = (raw_acc / 100.0) if raw_acc is not None else 1.0

    # base_power: null for status moves
    base_power = data.get("power") or 0

    # damage class: physical / special / status
    category = data.get("damage_class", {}).get("name", "physical")

    # type
    type_ = data.get("type", {}).get("name", "normal")

    # pp
    pp = data.get("pp") or 16

    # priority
    priority = data.get("priority", 0)

    # contact flag lives in move flags list
    flags = {f["name"] for f in data.get("meta", {}).get("flags", [])}
    # PokeAPI stores flags differently — check top-level "past_values" isn't it
    # The actual flags are under data["flags"] as a list of {name, url} dicts
    flag_names = {f["name"] for f in data.get("flags", [])}
    makes_contact = "contact" in flag_names

    return {
        "name": data["name"],          # canonical slug form
        "base_power": base_power,
        "accuracy": accuracy,
        "pp": pp,
        "type": type_,
        "category": category,
        "priority": priority,
        "makes_contact": makes_contact,
    }


def _prefetch_all_moves(rate_limit_delay: float = 0.05):
    """
    Fetch every move from PokeAPI in one go and populate the cache.
    Run this once with:  python replay_parser.py --prefetch-moves
    Takes a few minutes — PokeAPI has ~900 moves.
    """
    _load_disk_cache()

    # Get the full move list
    url = f"{POKEAPI_BASE}/move?limit=10000"
    req = urllib.request.Request(url, headers={"User-Agent": "pokemon-il-parser/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        index = json.loads(resp.read().decode("utf-8"))

    slugs = [entry["name"] for entry in index["results"]]
    already_cached = set(_MOVE_CACHE.keys())
    to_fetch = [s for s in slugs if s not in already_cached]

    print(f"[MoveDB] {len(already_cached)} moves already cached, "
          f"fetching {len(to_fetch)} more...")

    for i, slug in enumerate(to_fetch, 1):
        try:
            features = _fetch_move_from_api(slug)
            _MOVE_CACHE[slug] = features
            if i % 50 == 0:
                print(f"  [{i}/{len(to_fetch)}] fetched '{slug}' — saving checkpoint...")
                _save_disk_cache()
            time.sleep(rate_limit_delay)   # be polite to the API
        except Exception as e:
            print(f"  [WARN] Failed to fetch '{slug}': {e}")

    _save_disk_cache()
    print(f"[MoveDB] Done. {len(_MOVE_CACHE)} total moves cached to {_CACHE_PATH}")


def get_move_features(move_name: str, current_pp: int = None) -> dict:
    """
    Look up move features, hitting the in-memory cache first, then disk cache,
    then PokeAPI as a last resort. Falls back to neutral defaults on failure.
    """
    slug = move_name.lower().strip().replace(" ", "-")

    # 1. In-memory hit
    if slug in _MOVE_CACHE:
        features = dict(_MOVE_CACHE[slug])
        features["name"] = move_name          # preserve original casing
        features["current_pp"] = current_pp if current_pp is not None else features["pp"]
        return features

    # 2. Disk cache not yet loaded — try loading it
    if not _MOVE_CACHE:
        _load_disk_cache()
        if slug in _MOVE_CACHE:
            features = dict(_MOVE_CACHE[slug])
            features["name"] = move_name
            features["current_pp"] = current_pp if current_pp is not None else features["pp"]
            return features

    # 3. Live API fetch
    try:
        features = _fetch_move_from_api(slug)
        _MOVE_CACHE[slug] = features
        _save_disk_cache()                    # persist so we don't fetch again
        result = dict(features)
        result["name"] = move_name
        result["current_pp"] = current_pp if current_pp is not None else result["pp"]
        return result
    except Exception as e:
        print(f"[MoveDB] Warning: could not fetch '{slug}' from PokeAPI: {e}. "
              f"Using defaults. Run --prefetch-moves to pre-populate the cache.")

    # 4. Fallback defaults
    pp_default = 16
    return {
        "name": move_name,
        "base_power": 0,
        "accuracy": 1.0,
        "pp": pp_default,
        "current_pp": current_pp if current_pp is not None else pp_default,
        "type": "normal",
        "category": "physical",
        "priority": 0,
        "makes_contact": False,
    }


#Parser

WEATHER_TAGS = {"sunnyday", "raindance", "sandstorm", "hail", "snow", "desolateland",
                "primordialsea", "deltastream"}
TERRAIN_TAGS = {"electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"}
SIDE_CONDITIONS = {"stealthrock", "spikes", "toxicspikes", "stickyweb",
                   "reflect", "lightscreen", "auroraveil"}
STATUS_MAP = {
    "brn": "brn", "par": "par", "psn": "psn",
    "tox": "tox", "slp": "slp", "frz": "frz",
    "": None, "0": None,
}

def normalize_species(raw: str) -> str:
    """Strip nickname/form info: 'p1a: Garchomp' -> 'garchomp'"""
    if ": " in raw:
        raw = raw.split(": ", 1)[1]
    return raw.split(",")[0].strip().lower()

def normalize_move(raw: str) -> str:
    return raw.strip().lower().replace(" ", "-")

class BattleState: 
    """Mutable state machien that tacks one battle as we walk its log"""

    def __init__(self, battle_id:str, perspective: str = "p1"):
        self.battle_id = battle_id
        self.perspective = perspective  #whose POV to record
        self.turn = 0
        self.winner: Optional[str] = None

        #Each side: keyed by p1 and p2
        self.teams: dict[str, dict[str,PokemonState]] = {"p1": {}, "p2": {}}
        self.active: dict[str, Optional[str]] = {"p1": None, "p2": None}
        self.conditions: dict[str, list] = {"p1": [], "p2": []}

        self.weather: Optional[str] = None
        self.terrain: Optional[str] = None
        self.trick_room = False

        self.records: list[TurnRecord] = []

        #Buffer: actions seen this turn before we emit a record
        self._pending_actions: dict[str, tuple] = {}   # player -> (type, name)
        self._force_switch: dict[str, bool] = {"p1": False, "p2": False}
    
    #Helpers

    def _side(self, ident: str) -> str:
        """extract p1 or p2 from identifies like p1a: Garchomp"""
        return ident[:2]
    
    def _get_active(self, player: str) -> Optional[PokemonState]:
        name = self.active.get(player)
        if name:
            return self.teams[player].get(name)
        return None
    
    def _snapshot_pokemon(self, pmon: PokemonState, reveal_moves = True) -> dict:
        d = asdict(pmon)
        if not reveal_moves:
            #only returns moves that have been revealed (non-empy lists already filtered)
            pass
        return d
    
    def _snapshot_state(self, acting_player: str) -> dict:
        """
        Return a dict representing the full observable state from acting_player's POV.
        Opponent info is limited to what has been revealed.
        """
        opp = "p2" if acting_player == "p1" else "p1"

        my_active_pmon = self._get_active(acting_player)
        opp_active_pmon = self._get_active(opp)

        my_bench = [
            self._snapshot_pokemon(p)
            for name, p in self.teams[acting_player].items()
            if name != self.active[acting_player] and not p.fainted
        ]
        # Opponent bench: only revealed species, limited move info
        opp_bench = [
            {
                "species": p.species,
                "hp_frac": p.hp_frac,
                "fainted": p.fainted,
                "status": p.status,
                "moves": p.moves,   # only revealed moves
                "item": p.item,
                "ability": p.ability,
            }
            for name, p in self.teams[opp].items()
            if name != self.active[opp] and not p.fainted
        ]

        return {
            "my_active": self._snapshot_pokemon(my_active_pmon) if my_active_pmon else {},
            "my_bench": my_bench,
            "my_conditions": list(self.conditions[acting_player]),
            "opp_active": self._snapshot_pokemon(opp_active_pmon) if opp_active_pmon else {},
            "opp_bench": opp_bench,
            "opp_conditions": list(self.conditions[opp]),
            "weather": self.weather,
            "terrain": self.terrain,
            "trick_room": self.trick_room,
            "turn": self.turn,
        }

    def _available_moves(self, player: str) -> list:
        pmon = self._get_active(player)
        if pmon is None:
            return []
        return list(pmon.moves)   # list of move feature dicts

    def _available_switches(self, player: str) -> list:
        return [
            {"species": p.species, "hp_frac": p.hp_frac, "status": p.status,
             "moves": p.moves, "ability": p.ability, "item": p.item}
            for name, p in self.teams[player].items()
            if name != self.active[player] and not p.fainted
        ]
    def _emit_record(self, player: str, action_type: str, action_name: str,
                     force_switch: bool = False):
        snap = self._snapshot_state(player)
        avail_moves = self._available_moves(player)
        avail_switches = self._available_switches(player)

        # Compute slot (1-indexed position in available list)
        slot = 0
        if action_type == "move":
            for i, m in enumerate(avail_moves, 1):
                if m["name"].lower().replace(" ", "-") == normalize_move(action_name):
                    slot = i
                    break
        elif action_type == "switch":
            for i, s in enumerate(avail_switches, 1):
                if s["species"].lower() == action_name.lower():
                    slot = i
                    break

        record = TurnRecord(
            battle_id=self.battle_id,
            turn_number=self.turn,
            player=player,
            perspective=player,
            **snap,
            available_moves=avail_moves,
            available_switches=avail_switches,
            force_switch=force_switch,
            action_type=action_type,
            action_name=action_name,
            action_slot=slot,
        )
        self.records.append(record)

    #Log Line Handlers

    def handle_poke(self, parts: list):
        """
        |poke|p1|Garchomp, L79, M|
        Declares a team member at battle start (random battles reveal full team).
        """
        if len(parts) < 3:
            return
        player = parts[1]
        species_raw = parts[2].split(",")[0].strip().lower()
        level = 100
        m = re.search(r"L(\d+)", parts[2])
        if m:
            level = int(m.group(1))

        pmon = PokemonState(species=species_raw, level=level)
        self.teams[player][species_raw] = pmon

    def handle_switch(self, parts: list, is_drag=False):
        """
        |switch|p1a: Garchomp|Garchomp, L79, M|341/341
        """
        if len(parts) < 4:
            return
        player = self._side(parts[1])
        species = normalize_species(parts[1])
        hp_str = parts[3].split()[0]  # ignore status on same token

        # Parse HP
        if "/" in hp_str:
            cur, max_ = hp_str.split("/")
            hp_frac = int(cur) / int(max_) if int(max_) > 0 else 0.0
        else:
            hp_frac = 0.0 if hp_str == "0" else 1.0

        # Register pokemon if not seen before (opponent reveals)
        if species not in self.teams[player]:
            self.teams[player][species] = PokemonState(species=species)

        pmon = self.teams[player][species]
        pmon.hp_frac = hp_frac
        pmon.is_active = True

        # Deactivate previous active
        prev = self.active[player]
        if prev and prev in self.teams[player]:
            self.teams[player][prev].is_active = False

        self.active[player] = species
        self.teams[player][species].boosts = {
            "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0,
            "accuracy": 0, "evasion": 0
        }

        # If this was a forced switch action, emit the record now
        if self._force_switch[player] and not is_drag:
            self._emit_record(player, "switch", species, force_switch=True)
            self._force_switch[player] = False

    def handle_move(self, parts: list):
        """
        |move|p1a: Garchomp|Earthquake|p2a: Togekiss
        Emit state BEFORE applying effects (called during turn resolution).
        """
        if len(parts) < 3:
            return
        player = self._side(parts[1])
        move_name = parts[2].strip()
        norm = normalize_move(move_name)

        # Add move to the active pokemon's known move list if not already there
        pmon = self._get_active(player)
        if pmon:
            existing_names = {m["name"].lower().replace(" ", "-") for m in pmon.moves}
            if norm not in existing_names:
                pmon.moves.append(get_move_features(move_name))

        # Buffer this — we emit at turn boundary once we have both actions
        self._pending_actions[player] = ("move", move_name)
    def handle_damage(self, parts: list):
        """|-damage|p1a: Garchomp|210/341"""
        if len(parts) < 3:
            return
        player = self._side(parts[1])
        species = normalize_species(parts[1])
        hp_str = parts[2].split()[0]

        if species in self.teams[player]:
            if "/" in hp_str:
                cur, max_ = hp_str.split("/")
                self.teams[player][species].hp_frac = int(cur) / int(max_)
            elif hp_str == "0":
                self.teams[player][species].hp_frac = 0.0
                self.teams[player][species].fainted = True

    def handle_heal(self, parts: list):
        self.handle_damage(parts)   # same parsing logic

    def handle_status(self, parts: list):
        """|-status|p1a: Garchomp|brn"""
        if len(parts) < 3:
            return
        player = self._side(parts[1])
        species = normalize_species(parts[1])
        status = STATUS_MAP.get(parts[2].strip(), parts[2].strip())
        if species in self.teams[player]:
            self.teams[player][species].status = status

    def handle_curestatus(self, parts: list):
        """|-curestatus|p1a: Garchomp|brn"""
        if len(parts) < 2:
            return
        player = self._side(parts[1])
        species = normalize_species(parts[1])
        if species in self.teams[player]:
            self.teams[player][species].status = None
    
    def handle_boost(self, parts: list, negative=False):
        """|-boost|p1a: Garchomp|atk|2"""
        if len(parts) < 4:
            return
        player = self._side(parts[1])
        species = normalize_species(parts[1])
        stat = parts[2].strip()
        amount = int(parts[3].strip())
        if negative:
            amount = -amount
        if species in self.teams[player]:
            pmon = self.teams[player][species]
            if stat in pmon.boosts:
                pmon.boosts[stat] = max(-6, min(6, pmon.boosts[stat] + amount))


    def handle_weather(self, parts: list):
        """|-weather|SunnyDay  OR  |-weather|none"""
        if len(parts) < 2:
            return
        w = parts[1].strip().lower().replace(" ", "").replace("'", "")
        if w in ("none", ""):
            self.weather = None
        elif w in WEATHER_TAGS:
            self.weather = w

    def handle_fieldstart(self, parts: list):
        """|-fieldstart|move: Electric Terrain"""
        if len(parts) < 2:
            return
        effect = parts[1].lower().replace("move: ", "").replace(" ", "")
        if effect in TERRAIN_TAGS:
            self.terrain = effect
        elif effect == "trickroom":
            self.trick_room = not self.trick_room
    
    def handle_fieldend(self, parts: list):
        if len(parts) < 2:
            return
        effect = parts[1].lower().replace("move: ", "").replace(" ", "")
        if effect in TERRAIN_TAGS:
            self.terrain = None
        elif effect == "trickroom":
            self.trick_room = False
    
    def handle_sidestart(self, parts: list):
        """|sidestart|p1: Player|Stealth Rock"""
        if len(parts) < 3:
            return
        player = parts[1][:2]
        cond = parts[2].lower().replace("move: ", "").replace(" ", "")
        if cond in SIDE_CONDITIONS and cond not in self.conditions[player]:
            self.conditions[player].append(cond)
    
    def handle_sideend(self, parts: list):
        if len(parts) < 3:
            return
        player = parts[1][:2]
        cond = parts[2].lower().replace("move: ", "").replace(" ", "")
        if cond in self.conditions[player]:
            self.conditions[player].remove(cond)

    def handle_faint(self, parts: list):
        """
        |faint|p1a: Garchomp
        Mark as fainted; flag that this player needs a force switch next.
        """
        if len(parts) < 2:
            return
        player = self._side(parts[1])
        species = normalize_species(parts[1])
        if species in self.teams[player]:
            self.teams[player][species].fainted = True
            self.teams[player][species].hp_frac = 0.0
        # Only flag force switch if they have remaining pokemon
        remaining = [p for p in self.teams[player].values() if not p.fainted]
        if remaining:
            self._force_switch[player] = True
    
    def handle_turn(self, parts: list):
        """
        |turn|N  — emits buffered pending actions from the previous turn.
        """
        # Emit records for both players' actions from the turn that just ended
        for player, (atype, aname) in self._pending_actions.items():
            self._emit_record(player, atype, aname)
        self._pending_actions.clear()

        self.turn = int(parts[1]) if len(parts) > 1 else self.turn + 1

    def handle_win(self, parts: list):
        if len(parts) > 1:
            self.winner = parts[1].strip()
        # Flush any remaining pending actions
        for player, (atype, aname) in self._pending_actions.items():
            self._emit_record(player, atype, aname)
        self._pending_actions.clear()

    def handle_item(self, parts: list):
        """|-item|p1a: Garchomp|Rocky Helmet"""
        if len(parts) < 3:
            return
        player = self._side(parts[1])
        species = normalize_species(parts[1])
        item = parts[2].strip().lower()
        if species in self.teams[player]:
            self.teams[player][species].item = item
    
    def handle_ability(self, parts: list):
        """|-ability|p1a: Garchomp|Rough Skin"""
        if len(parts) < 3:
            return
        player = self._side(parts[1])
        species = normalize_species(parts[1])
        ability = parts[2].strip().lower().replace(" ", "-")
        if species in self.teams[player]:
            self.teams[player][species].ability = ability


#TOP LEVEL

HANDLERS = {
    "poke":        "handle_poke",
    "switch":      "handle_switch",
    "drag":        lambda s, p: s.handle_switch(p, is_drag=True),
    "move":        "handle_move",
    "-damage":     "handle_damage",
    "-heal":       "handle_heal",
    "-status":     "handle_status",
    "-curestatus": "handle_curestatus",
    "-boost":      "handle_boost",
    "-unboost":    lambda s, p: s.handle_boost(p, negative=True),
    "-weather":    "handle_weather",
    "-fieldstart": "handle_fieldstart",
    "-fieldend":   "handle_fieldend",
    "sidestart":   "handle_sidestart",
    "sideend":     "handle_sideend",
    "faint":       "handle_faint",
    "turn":        "handle_turn",
    "win":         "handle_win",
    "-item":       "handle_item",
    "-ability":    "handle_ability",
}

def parse_log(log_text: str, battle_id: str = None) -> list[dict]:
    """
    Parse a single Showdown replay log and return a list of turn record dicts.
    Each dict is one (state, action) training example.
    """
    if battle_id is None:
        battle_id = str(uuid.uuid4())[:8]

    state = BattleState(battle_id=battle_id)
    lines = log_text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue

        parts = line.split("|")
        # parts[0] is empty string before leading |
        parts = parts[1:]
        if not parts:
            continue

        tag = parts[0]

        if tag in HANDLERS:
            handler = HANDLERS[tag]
            if callable(handler):
                handler(state, parts)
            else:
                getattr(state, handler)(parts)

    # Label all records with the winner
    for record in state.records:
        record.winner = state.winner

    return [asdict(r) for r in state.records]


def parse_log_file(path: Path) -> list[dict]:
    """Parse a log file. Supports .log, .txt, and .json (Showdown JSON format)."""
    text = path.read_text(encoding="utf-8", errors="replace")

    # Showdown JSON replay format
    if path.suffix == ".json":
        try:
            data = json.loads(text)
            log_text = data.get("log", "")
            battle_id = data.get("id", path.stem)
            return parse_log(log_text, battle_id=battle_id)
        except json.JSONDecodeError:
            pass

    return parse_log(text, battle_id=path.stem)


#DATASET BUILDER

def build_dataset(log_dir: str, out_path: str, fmt: str = "jsonl",
                  min_turns: int = 3, players: list = None):
    """
    Walk a directory of replay files, parse each one, and write a dataset.

    Args:
        log_dir:    Directory containing .log / .txt / .json replay files
        out_path:   Output file path (.jsonl or .json)
        fmt:        "jsonl" (one JSON object per line) or "json" (array)
        min_turns:  Skip battles with fewer than this many turns (forfeits, etc.)
        players:    If set, only record turns for these player IDs (e.g. ["p1"])
    """
    log_dir = Path(log_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extensions = {".log", ".txt", ".json"}
    files = [f for f in log_dir.rglob("*") if f.suffix in extensions]

    all_records = []
    skipped = 0
    total_battles = 0

    print(f"Found {len(files)} replay files in {log_dir}")

    for fpath in files:
        try:
            records = parse_log_file(fpath)
        except Exception as e:
            print(f"  [WARN] Failed to parse {fpath.name}: {e}")
            skipped += 1
            continue

        if not records:
            skipped += 1
            continue

        # Filter by minimum turns
        turns_in_battle = max((r["turn_number"] for r in records), default=0)
        if turns_in_battle < min_turns:
            skipped += 1
            continue

        # Filter by player perspective if requested
        if players:
            records = [r for r in records if r["player"] in players]

        total_battles += 1
        all_records.extend(records)

    print(f"Parsed {total_battles} battles → {len(all_records)} turn records "
          f"({skipped} skipped)")

    # Write output
    if fmt == "jsonl":
        with open(out_path, "w") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")
    else:
        with open(out_path, "w") as f:
            json.dump(all_records, f, indent=2)

    print(f"Dataset written to {out_path}")
    return all_records


#CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Showdown replay logs into turn-level ML training data"
    )
    # --- Move cache ---
    parser.add_argument(
        "--prefetch-moves", action="store_true",
        help="Fetch all ~900 moves from PokeAPI and save to cache, then exit. "
             "Run this once before parsing large datasets."
    )
    parser.add_argument(
        "--cache", default="move_cache.json",
        help="Path to the move cache JSON file (default: move_cache.json)"
    )
    # --- Parsing ---
    parser.add_argument("--logs",
                        help="Directory containing replay log files")
    parser.add_argument("--out",
                        help="Output file path (e.g. dataset.jsonl)")
    parser.add_argument("--format", choices=["jsonl", "json"], default="jsonl",
                        help="Output format (default: jsonl)")
    parser.add_argument("--min-turns", type=int, default=3,
                        help="Skip battles shorter than this many turns (default: 3)")
    parser.add_argument("--player", choices=["p1", "p2", "both"], default="both",
                        help="Which player's turns to record (default: both)")
    args = parser.parse_args()

    # Point the move DB at the chosen cache file
    set_cache_path(Path(args.cache))

    if args.prefetch_moves:
        _prefetch_all_moves()
    else:
        if not args.logs or not args.out:
            parser.error("--logs and --out are required when not using --prefetch-moves")
        players = None if args.player == "both" else [args.player]
        build_dataset(
            log_dir=args.logs,
            out_path=args.out,
            fmt=args.format,
            min_turns=args.min_turns,
            players=players,
        )
