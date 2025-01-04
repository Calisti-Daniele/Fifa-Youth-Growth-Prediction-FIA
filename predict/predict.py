import os
import pickle
import numpy as np
import pandas as pd
from training_models.functions import *

# Percorsi ai file
dataset_path = '../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv'
dataset_fc_24_path = '../datasets/dataset_fc_24.csv'
models_dir = '../models/'

# Carica il dataset principale
df = load_dataset(dataset_path)
df = df.sort_values(by=['long_name', 'fifa_version'])

# Carica il dataset 2024
df_fc_24 = pd.read_csv(dataset_fc_24_path)

# Aggiungi le colonne 'experience' e 'age_trend' al dataset 2024
df_fc_24['experience'] = df_fc_24.groupby('long_name').cumcount() + 1
df_fc_24['age_trend'] = df_fc_24.groupby('long_name')['Age'].diff().fillna(0)

# Elenco dei target e relative feature
features = {
    'overall': ['potential', 'passing', 'dribbling', 'movement_reactions', 'mentality_composure'],
    'shooting': ['passing', 'dribbling', 'attacking_finishing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
                 'skill_long_passing', 'skill_ball_control', 'movement_agility', 'power_shot_power', 'power_long_shots',
                 'mentality_positioning', 'mentality_vision', 'mentality_penalties'],
    'passing': ['shooting', 'dribbling', 'attacking_crossing', 'attacking_short_passing', 'skill_dribbling',
                'skill_curve', 'skill_long_passing', 'skill_fk_accuracy', 'skill_ball_control', 'power_long_shots',
                'mentality_vision', 'mentality_positioning'],
    'dribbling': ['shooting', 'passing', 'attacking_crossing', 'attacking_finishing', 'attacking_volleys',
                  'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_ball_control', 'movement_acceleration',
                  'movement_agility', 'movement_balance', 'power_long_shots', 'mentality_positioning',
                  'mentality_vision'],
    'defending': ['physic', 'mentality_aggression', 'mentality_interceptions', 'defending_marking_awareness',
                  'defending_standing_tackle', 'defending_sliding_tackle'],
    'physic': ['defending', 'power_strength', 'mentality_aggression', 'mentality_interceptions']
}

timesteps = 5  # Lunghezza della sequenza temporale

# Funzione per effettuare la previsione e creare il dataset
all_predictions = []

def predict_and_collect_results(target, feature_columns):
    print(f"Inizio predizioni per il target: {target}")

    # Percorsi dei modelli e scaler
    model_path = os.path.join(models_dir, target, f"{target}_model.keras")
    scaler_X_path = os.path.join(models_dir, target, "scaler_X.pkl")
    scaler_y_path = os.path.join(models_dir, target, "scaler_y.pkl")

    # Carica il modello e gli scaler
    model = keras.models.load_model(model_path, custom_objects={"weighted_loss": weighted_loss})
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    # Prepara i dati per il test
    X, player_names = [], []
    for player, player_data in df.groupby('long_name'):
        if len(player_data) >= timesteps + 1:
            player_data = player_data[feature_columns].values
            X.append(player_data[-timesteps:, :])  # Prendi le ultime timesteps
            player_names.append(player)  # Salva il nome del giocatore

    X = np.array(X)  # Shape: (num_players, timesteps, num_features)
    print(X.shape)

    # Normalizza i dati
    X_flat = X.reshape(-1, len(feature_columns))
    X_scaled = scaler_X.transform(X_flat).reshape(X.shape)

    # Predizioni
    y_pred_scaled = model.predict(X_scaled)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

    # Aggiungi i risultati alle predizioni
    for player, pred_value in zip(player_names, y_pred_original.flatten()):
        all_predictions.append({
            'Nome': player,
            target: pred_value
        })

# Effettua le predizioni per ogni target
for target_name, feature_cols in features.items():
    predict_and_collect_results(target_name, feature_cols)

# Crea un DataFrame con tutte le predizioni
predictions_df = pd.DataFrame(all_predictions)
predictions_df = predictions_df.groupby('Nome').first().reset_index()

# Raggruppa il dataset principale per long_name e tieni solo la prima riga
df = df.groupby('long_name').first().reset_index()

# Merge con il dataset principale
merged_df = pd.merge(predictions_df,
                     df[['long_name','player_url', 'short_name', 'player_positions', 'nationality_name', 'preferred_foot']].drop_duplicates(),
                     left_on='Nome', right_on='long_name', how='left')

# Salva il dataset finale
output_path = 'fifa_predictions_with_metadata.csv'
merged_df.to_csv(output_path, index=False)
print(f"Dataset finale salvato in '{output_path}'")

