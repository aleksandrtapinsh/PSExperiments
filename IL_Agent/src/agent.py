import json
import requests
from pathlib import Path


""" 
Cleans dataset 
Fills in all missing base power values for moves 
"""
moves_db = requests.get("https://play.pokemonshowdown.com/data/moves.json").json()

def normalize(name):
    return name.lower().replace(" ", "").replace("-", "")

def get_move_base_power(move):
    data = moves_db.get(normalize(move), {})
    print(data)
    bp = data.get("basePower", 0)
    accuracy = data.get("accuracy", 1.0)

    if accuracy is True:
        accuracy = 1.0
    else: 
        accuracy = accuracy/100

    return bp, accuracy

def enrich_moves(move_list):
    enriched = []
    for m in move_list:
        bp, acc = get_move_base_power(m["name"])
        enriched.append({**m, "base_power": bp, "accuracy": acc})
    return enriched

def enrich_record(record):
    record["my_active"]["moves"] = enrich_moves(record["my_active"]["moves"])
    record["opp_active"]["moves"] = enrich_moves(record["opp_active"]["moves"])
    record["available_moves"] = enrich_moves(record["available_moves"])

    for p in record["my_bench"]:
        p["moves"] = enrich_moves(p.get("moves", []))
    for p in record["opp_bench"]:
        p["moves"] = enrich_moves(p.get("moves", []))

    return record

BASE = Path(__file__).parent.parent
input_path  = BASE / "parser" / "dataset.jsonl"
output_path = BASE / "parser" / "dataset_with_bp.jsonl"

with open(input_path) as fin, open(output_path, "w") as fout:
    for line in fin:
        record = json.loads(line.strip())
        enriched = enrich_record(record)
        fout.write(json.dumps(enriched) + "\n")

print("Done!")

""" Starting IL Training here after data cleaning """

def get_battle_log(filepath):
    with open(filepath) as f:
        data = json.load(f)

        state_action_pairs = []
        for turn in data["turn_number"]:

            state = extract_state(turn)
            action = extract_action(turn)
            state_action_pairs.append((state, action))
    return state_action_pairs  

def extract_state(turn):
    return "TODO"

def extract_action(turn):
    return "TODO"