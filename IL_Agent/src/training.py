import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# Load data
BASE = Path(__file__).parent.parent
file_path = BASE / "parser" / "cleaned_dataset.jsonl"

STATUSES  = [None, "brn", "par", "slp", "frz", "psn", "tox"]
WEATHERS  = [None, "raindance", "sunnyday", "sandstorm", "hail", "snow"]
TERRAINS  = [None, "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]
TYPES     = ["Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison",
             "Ground","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy"]
CATEGORIES = ["physical", "special", "status"]

MAX_MOVES = 4
MAX_BENCH = 5

MOVE_DIM = 3 + len(TYPES) + len(CATEGORIES)
POKEMON_DIM = 2 + len(STATUSES) + 7 + MAX_MOVES * MOVE_DIM
BENCH_DIM = MAX_BENCH * POKEMON_DIM

data_list = []
try:
    with open(file_path, 'r') as file:
        print("Loading data...")
        for line in file:
            json_object = json.loads(line)
            data_list.append(json_object)
except FileNotFoundError:
    print(f"{file_path} not found.")
except json.JSONDecodeError:
    print("Error decoding the json")

# One hot encoding helper
def one_hot(val, options):
    vector = np.zeros(len(options))
    if val in options:
        vector[options.index(val)] = 1.0
    return vector

def normalize_name(name):
    return name.lower().strip()

def encode_move(move):
    if move is None:
        return np.zeros(MOVE_DIM)
    return np.concatenate([
        [move.get("base_power", 0) / 250],
        [move.get("accuracy", 1.0)],
        [move.get("current_pp", 0) / max(move.get("pp", 1), 1)],
        one_hot(move.get("type"), TYPES),
        one_hot(move.get("category"), CATEGORIES),
    ])

def encode_pokemon(mon):
    if mon is None:
        return np.zeros(POKEMON_DIM)

    moves = mon.get("moves", [])
    move_vecs = [encode_move(m) for m in moves[:MAX_MOVES]]
    while len(move_vecs) < MAX_MOVES:
        move_vecs.append(np.zeros(MOVE_DIM))

    boosts = mon.get("boosts", {})
    boost_vec = np.array([
        boosts.get("atk", 0) / 6,
        boosts.get("def", 0) / 6,
        boosts.get("spa", 0) / 6,
        boosts.get("spd", 0) / 6,
        boosts.get("spe", 0) / 6,
        boosts.get("accuracy", 0) / 6,
        boosts.get("evasion", 0) / 6,
    ])

    return np.concatenate([
        [mon.get("hp_frac", 0.0)],
        [float(mon.get("fainted", False))],
        one_hot(mon.get("status"), STATUSES),
        boost_vec,
        *move_vecs,
    ])

def encode_bench(bench):
    vecs = [encode_pokemon(p) for p in bench[:MAX_BENCH]]
    while len(vecs) < MAX_BENCH:
        vecs.append(np.zeros(POKEMON_DIM))
    return np.concatenate(vecs)

def encode_battle(turn):
    return np.concatenate([
        encode_pokemon(turn["my_active"]),
        encode_bench(turn["my_bench"]),
        encode_pokemon(turn["opp_active"]),
        encode_bench(turn["opp_bench"]),
        one_hot(turn.get("weather"), WEATHERS),
        one_hot(turn.get("terrain"), TERRAINS),
        [float(turn.get("trick_room", False))],
        [min(turn.get("turn_number", 0), 100) / 100],
    ]).astype(np.float32)

def encode_move_action(turn):
    chosen = normalize_name(turn["action_name"])
    
    # check available_moves
    for i, m in enumerate(turn["available_moves"]):
        if normalize_name(m["name"]) == chosen:
            return i
    
    # covers pivot moves
    for i, m in enumerate(turn["my_active"].get("moves", [])):
        if normalize_name(m["name"]) == chosen:
            return i
    
    raise ValueError(f"Failed to move with: {chosen}")

def encode_switch_action(turn):
    chosen = normalize_name(turn["action_name"])
    
    # check available_switches
    for i, s in enumerate(turn["available_switches"]):
        if normalize_name(s["species"]) == chosen:
            return i
    
    # check my_bench
    for i, s in enumerate(turn.get("my_bench", [])):
        if normalize_name(s["species"]) == chosen:
            return i
    
    raise ValueError(f"Failed to switch to: {chosen}")

def move_mask(turn):
    mask = np.zeros(4, dtype=np.float32)
    for i in range(min(len(turn["available_moves"]), 4)):
        mask[i] = 1.0
    return mask

def switch_mask(turn):
    mask = np.zeros(5, dtype=np.float32)
    for i in range(min(len(turn["available_switches"]), 5)):
        mask[i] = 1.0
    return mask

# Main vectorization #
def vectorize_turns(turn_list):
    move_states, move_actions, move_masks     = [], [], []
    switch_states, switch_actions, switch_masks = [], [], []

    for turn in turn_list:
        state = encode_battle(turn)

        if turn["action_type"] == "move":
            try:
                action = encode_move_action(turn)
            except ValueError as e:
                print(f"Skipping turn: {e}")
                continue
            move_states.append(state)
            move_actions.append(action)
            move_masks.append(move_mask(turn))

        else:  # switch
            try:
                action = encode_switch_action(turn)
            except ValueError as e:
                print(f"Skipping turn: {e}")
                continue
            switch_states.append(state)
            switch_actions.append(action)
            switch_masks.append(switch_mask(turn))

    move_data = (
        np.array(move_states,  dtype=np.float32),
        np.array(move_actions, dtype=np.int32),
        np.array(move_masks,   dtype=np.float32),
    )
    switch_data = (
        np.array(switch_states,  dtype=np.float32),
        np.array(switch_actions, dtype=np.int32),
        np.array(switch_masks,   dtype=np.float32),
    )
    return move_data, switch_data

# --- Load and Run ---
def load_and_vectorize(path):
    turns = []
    with open(path) as f:
        for line in f:
            turns.append(json.loads(line.strip()))

    move_data, switch_data = vectorize_turns(turns)
    move_states, move_actions, move_masks     = move_data
    switch_states, switch_actions, switch_masks = switch_data

    print(f"Move turns:   {len(move_actions)} | state shape: {move_states.shape}")
    print(f"Switch turns: {len(switch_actions)} | state shape: {switch_states.shape}")
    return move_data, switch_data

move_data, switch_data = load_and_vectorize(
    Path(__file__).parent.parent / "parser" / "cleaned_dataset.jsonl"
)