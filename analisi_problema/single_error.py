import pickle
import numpy as np
import pandas as pd
import keras

from training_models.functions import load_dataset
from training_models.testing_training import weighted_loss

# Percorsi ai file salvati
dataset_path = '../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv'
dataset_fc_24_path = '../datasets/dataset_fc_24.csv'
model_path = '../models/fia_model.keras'
scaler_X_path = '../models/scaler_X.pkl'
scaler_y_path = '../models/scaler_y.pkl'
params_path = '../models/model_params.pkl'

# Carica il dataset principale
df = load_dataset(dataset_path)

# Aggiungi le colonne 'experience' e 'age_trend'
df = df.sort_values(by=['long_name', 'fifa_version'])


# Carica il modello salvato
model = keras.models.load_model(model_path, custom_objects={"weighted_loss": weighted_loss})

# Carica gli scaler
with open(scaler_X_path, 'rb') as f:
    scaler_X = pickle.load(f)
with open(scaler_y_path, 'rb') as f:
    scaler_y = pickle.load(f)

# Carica i parametri
with open(params_path, 'rb') as f:
    params = pickle.load(f)

# Estrai i parametri
features = params['features']
timesteps = params['timesteps']

# Prepara i dati per il test
X, player_names = [], []
for player, player_data in df.groupby('long_name'):
    if len(player_data) >= timesteps:
        player_data = player_data[features].values
        X.append(player_data[-timesteps:, :])  # Prendi le ultime `timesteps`
        player_names.append(player)  # Salva il nome del giocatore

X = np.array(X)  # Shape: (num_players, timesteps, num_features)

# Normalizza i dati
X_flat = X.reshape(-1, len(features))
X_scaled = scaler_X.transform(X_flat).reshape(X.shape)

# Predizioni
y_pred_scaled = model.predict(X_scaled)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

# Carica il dataset 2024
df_fc_24 = pd.read_csv(dataset_fc_24_path)

# Aggiungi le colonne 'experience' e 'age_trend' anche al dataset 2024
df_fc_24['experience'] = df_fc_24.groupby('long_name').cumcount() + 1
df_fc_24['age_trend'] = df_fc_24.groupby('long_name')['Age'].diff().fillna(0)

# Confronta le predizioni con i valori effettivi del dataset 2024
results = []
for player, pred_value in zip(player_names, y_pred_original.flatten()):
    # Valore reale dal dataset 2024
    actual_value = df_fc_24[df_fc_24['long_name'] == player]['Defending'].values
    if actual_value.size > 0:  # Verifica che il valore esista
        actual_value = actual_value[0]
    else:
        actual_value = None  # Il valore non è disponibile nel dataset 2024

    # Aggiungi al risultato
    results.append({
        'Nome': player,
        'Previsto': pred_value,
        'Effettivo (2024)': actual_value,
        'Differenza': abs(pred_value - actual_value) if actual_value is not None else None
    })

# Crea un DataFrame per il confronto
comparison_df = pd.DataFrame(results)

# Ordina per differenza decrescente
comparison_df = comparison_df.sort_values(by='Differenza', ascending=False)

# Mostra i primi 10 risultati
print(comparison_df.head(10))

# Salva il DataFrame per ulteriori analisi (opzionale)
comparison_df.to_csv('outputs/fia_comparison_2024.csv', index=False)
print("Confronto salvato in 'outputs/fia_comparison_2024.csv'")
