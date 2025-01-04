from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import os
from training_models.functions import *


# Percorsi dei file
models_dir = '../models/'
dataset_path = '../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv'
output_dir = 'outputs/'

# Carica il dataset
print("Caricamento del dataset...")
df = load_dataset(dataset_path)
df = df.sort_values(by=['long_name', 'fifa_version'])

# Aggiunta di feature extra
df['experience'] = df.groupby('long_name').cumcount() + 1
df['age_trend'] = df.groupby('long_name')['age'].diff().fillna(0)

# Definizione dei target e feature
features_dict = {
    'overall': ['potential', 'passing', 'dribbling', 'movement_reactions', 'mentality_composure'],
    'potential': ['overall', 'passing', 'dribbling'],
    'shooting': ['passing', 'dribbling', 'attacking_finishing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
                 'skill_long_passing', 'skill_ball_control', 'movement_agility', 'power_shot_power', 'power_long_shots',
                 'mentality_positioning', 'mentality_vision', 'mentality_penalties'],
    'passing': ['shooting', 'dribbling', 'attacking_crossing', 'attacking_short_passing', 'skill_dribbling',
                'skill_curve',
                'skill_long_passing', 'skill_fk_accuracy', 'skill_ball_control', 'power_long_shots', 'mentality_vision',
                'mentality_positioning'],
    'dribbling': ['shooting', 'passing', 'attacking_crossing', 'attacking_finishing', 'attacking_volleys',
                  'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_ball_control', 'movement_acceleration',
                  'movement_agility', 'movement_balance', 'power_long_shots', 'mentality_positioning',
                  'mentality_vision'],
    'defending': ['physic', 'mentality_aggression', 'mentality_interceptions', 'defending_marking_awareness',
                  'defending_standing_tackle', 'defending_sliding_tackle'],
    'physic': ['defending', 'power_strength', 'mentality_aggression', 'mentality_interceptions']
}

timesteps = 5
num_folds = 5


# Funzione principale per valutare ogni target con TimeSeriesSplit
def evaluate_model_with_tscv(target_name, features, model_path, scaler_X_path, scaler_y_path):
    print(f"\nInizio valutazione per il target: {target_name}")

    # Carica il modello e gli scaler
    model = keras.models.load_model(model_path)
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    # Prepara i dati
    X, y = [], []
    for player, player_data in df.groupby('long_name'):
        if len(player_data) >= timesteps + 1:
            player_data_values = player_data[features + [target_name]].values
            for i in range(len(player_data_values) - timesteps):
                X.append(player_data_values[i:i + timesteps, :-1])  # Input
                y.append(player_data_values[i + timesteps, -1])  # Target

    X = np.array(X)
    y = np.array(y)

    # Normalizza i dati
    X_flat = X.reshape(-1, len(features))
    X_scaled = scaler_X.transform(X_flat).reshape(X.shape)
    y_scaled = scaler_y.transform(y.reshape(-1, 1))

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=num_folds)
    metrics = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):
        print(f"\nFold {fold + 1}/{num_folds}")
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_scaled[train_index], y_scaled[test_index]

        # Valutazione
        loss, mae = model.evaluate(X_test, y_test, verbose=1)
        metrics.append({'loss': loss, 'mae': mae})

    # Media e deviazione standard
    avg_loss = np.mean([m['loss'] for m in metrics])
    avg_mae = np.mean([m['mae'] for m in metrics])
    std_loss = np.std([m['loss'] for m in metrics])
    std_mae = np.std([m['mae'] for m in metrics])

    print(f"\nRisultati per {target_name}:")
    print(f"Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    return target_name, avg_loss, std_loss, avg_mae, std_mae


# Ciclo su tutti i target
all_results = []
for target_name, feature_columns in features_dict.items():
    model_dir = os.path.join(models_dir, target_name)
    results = evaluate_model_with_tscv(
        target_name=target_name,
        features=feature_columns,
        model_path=os.path.join(model_dir, f"{target_name}_model.keras"),
        scaler_X_path=os.path.join(model_dir, "scaler_X.pkl"),
        scaler_y_path=os.path.join(model_dir, "scaler_y.pkl")
    )
    all_results.append(results)

# Stampa finale di tutti i risultati
print("\n\nRiepilogo finale dei risultati:")
for result in all_results:
    target_name, avg_loss, std_loss, avg_mae, std_mae = result
    print(f"Target: {target_name}")
    print(f"Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}\n")
