import json
import requests
from pathlib import Path

""" 
Cleans dataset 
Fills in all missing base power values for moves 
"""
moves_db = requests.get("https://play.pokemonshowdown.com/data/moves.json").json()
pokemon_db = requests.get("https://play.pokemonshowdown.com/data/pokedex.json").json()

def normalize(name):
    return name.lower().replace(" ", "").replace("-", "")

def get_pokemon_type(pokemon):
    data = pokemon_db.get(pokemon, {})
    pokemon_type = data.get("types", [])
    type1 = pokemon_type[0] if len(pokemon_type) > 0 else None
    type2 = pokemon_type[1] if len(pokemon_type) > 1 else None
    print(pokemon_type)

    return type1, type2

def get_move_data(move):
    data = moves_db.get(normalize(move), {})
    bp = data.get("basePower", 0)
    move_type = data.get("type", 0)
    accuracy = data.get("accuracy", 1.0)
    if accuracy is True:
        accuracy = 1.0
    else: 
        accuracy = accuracy/100

    return bp, accuracy, move_type

def enrich_moves(move_list):
    enriched = []
    for m in move_list:
        bp, acc, move_type = get_move_data(m["name"])
        enriched.append({**m, "base_power": bp, "accuracy": acc, "type": move_type})
    return enriched

def enrich_pokemon(pokemon_list):
    enriched = []
    for p in pokemon_list:
        type1, type2 = get_pokemon_type(p["species"])
        enriched.append({**p, "type1": type1, "type2": type2})
    return enriched

def enrich_record(record):
    record["my_active"]["moves"]   = enrich_moves(record["my_active"]["moves"])
    record["opp_active"]["moves"]  = enrich_moves(record["opp_active"]["moves"])
    record["available_moves"]      = enrich_moves(record["available_moves"])

    record["my_active"]  = enrich_pokemon([record["my_active"]])[0]
    record["opp_active"] = enrich_pokemon([record["opp_active"]])[0]
    record["my_bench"]   = enrich_pokemon(record["my_bench"])
    record["opp_bench"]  = enrich_pokemon(record["opp_bench"])

    for p in record["my_bench"]:
        p["moves"] = enrich_moves(p.get("moves", []))
    for p in record["opp_bench"]:
        p["moves"] = enrich_moves(p.get("moves", []))

    return record

def clean_data():
    BASE = Path(__file__).parent.parent
    input_path  = BASE / "parser" / "dataset.jsonl"
    output_path = BASE / "parser" / "cleaned_dataset.jsonl"

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            record = json.loads(line.strip())
            enriched = enrich_record(record)
            fout.write(json.dumps(enriched) + "\n")

    print("Done!")

clean_data()
