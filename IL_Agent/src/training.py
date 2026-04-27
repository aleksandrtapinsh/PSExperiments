import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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
    for i, m in enumerate(turn["available_moves"][:4]):
        if (i > 3):
            print(f"available moves: {i}")
        if normalize_name(m["name"]) == chosen:
            return i
    
    # covers pivot moves
    for i, m in enumerate(turn["my_active"].get("moves", [])[:4]):
        if (i > 3):
            print(f"active moves: {i}")
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

def evaluate_model(model, X, masks, y, name="Model"):
    probs = model.predict([X, masks], verbose=0)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y, preds)
    f1_weighted = f1_score(y, preds, average="weighted")
    f1_macro = f1_score(y, preds, average="macro")
    cm = confusion_matrix(y, preds)

    print(f"\n{name} Evaluation")
    print("-" * 40)
    print(f"Accuracy:      {acc:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"F1 (macro):    {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)
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
                #print(f"Skipping turn: {e}")
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

import tensorflow as tf
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).parent.parent / "models"

def build_rf_model(X_train, y_train):
    random_forest = RandomForestClassifier(n_estimators=200, 
                                           max_depth=15, 
                                           class_weight='balanced',
                                           min_samples_split=5, 
                                           min_samples_leaf=5,
                                           n_jobs=-1, 
                                           random_state=42)

    random_forest.fit(X_train, y_train)
    return random_forest

def build_nn_model(state_dim, num_actions):
    """Shared architecture for both move and switch models."""
    inputs = tf.keras.Input(shape=(state_dim,))
    mask   = tf.keras.Input(shape=(num_actions,))

    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    logits = tf.keras.layers.Dense(num_actions)(x)

    # Mask illegal actions before softmax
    masked = logits + (1.0 - mask) * -1e9
    output = tf.keras.layers.Softmax()(masked)

    return tf.keras.Model(inputs=[inputs, mask], outputs=output)


move_data, switch_data = load_and_vectorize(
    Path(__file__).parent.parent / "parser" / "cleaned_dataset.jsonl"
)
move_states, move_actions, move_masks     = move_data
switch_states, switch_actions, switch_masks = switch_data

state_dim = move_states.shape[1]
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Move models ---

# Neural Network
move_model = build_nn_model(state_dim, num_actions=4)
move_model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)
history = move_model.fit(
    [move_states, move_masks],
    move_actions,
    epochs=12,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

# Plot move NN model
plt.figure(figsize=(10,4))
#Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Move Model Loss')
# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Move Model Accuracy')
plt.show()

evaluate_model(move_model, move_states, move_masks, move_actions, "Move Model")
move_model.save(str(MODEL_DIR / "move_model.keras"))
print(f"Move model saved → {MODEL_DIR / 'move_model.keras'}")

# Random Forest
X_train_move, X_test_move, y_train_move, y_test_move = train_test_split(
    move_states, move_actions, test_size=0.2, random_state=42
)

rf_move = build_rf_model(X_train_move, y_train_move)

# Evaluate rf_move
y_pred_move = rf_move.predict(X_test_move)
accuracy = accuracy_score(y_test_move, y_pred_move)

print(f"Model accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_move, y_pred_move, labels=rf_move.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Save random forest move model
with open(MODEL_DIR / 'rf_move_model.pk1', 'wb') as f:
    pickle.dump(rf_move, f)
    print("Saved rf_move_model.pk1")

# --- Switch model ---
if len(switch_actions) > 0:
    switch_model = build_nn_model(state_dim, num_actions=5)
    switch_model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    switch_history = switch_model.fit(
        [switch_states, switch_masks],
        switch_actions,
        epochs=12,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    # Plot switch NN model
    plt.figure(figsize=(10,4))
    #Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(switch_history.history['loss'], label='train_loss')
    plt.plot(switch_history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Switch Model Loss')
    #Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(switch_history.history['accuracy'], label='train_acc')
    plt.plot(switch_history.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Switch Model Accuracy')
    plt.show()
    evaluate_model(switch_model, switch_states, switch_masks, switch_actions, "Switch Model")
    switch_model.save(str(MODEL_DIR / "switch_model.keras"))
    print(f"Switch model saved → {MODEL_DIR / 'switch_model.keras'}")
else:
    print("No switch turns found in dataset — switch model not trained.")

# Random Forest
X_train_switch, X_test_switch, y_train_switch, y_test_switch = train_test_split(
    switch_states, switch_actions, test_size=0.2, random_state=42
)

rf_switch = build_rf_model(X_train_switch, y_train_switch)

# Evaluate rf_switch
y_pred_switch = rf_switch.predict(X_test_switch)
accuracy = accuracy_score(y_test_switch, y_pred_switch)


print(f"Model accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_switch, y_pred_switch, labels=rf_switch.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Save random forest switch model
with open(MODEL_DIR / 'rf_switch_model.pk1', 'wb') as f:
    pickle.dump(rf_switch, f)
    print("Saved rf_switch_model.pk1")