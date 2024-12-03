import keras
import numpy as np
import pandas as pd
import pickle

# Carica il modello salvato
model_path = '../models/fia_model.keras'
model = keras.models.load_model(model_path)

# Carica gli scaler
with open('../models/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('../models/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Carica i parametri
with open('../models/model_params.pkl', 'rb') as f:
    params = pickle.load(f)

# Estrai i parametri
features = params['features']
timesteps = params['timesteps']

# Carica il dataset
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv')


def get_player_data(player_name):
    """Funzione per ottenere i dati del giocatore con il nome specificato"""
    player_data = df[df['long_name'] == player_name]
    if len(player_data) >= timesteps + 1:
        return player_data
    else:
        return None  # Se il giocatore non ha abbastanza dati


def predict_defending(player_name):
    # Recupera i dati del giocatore
    player_data = get_player_data(player_name)
    if player_data is None:
        print(f"Il giocatore {player_name} non ha abbastanza dati per fare una previsione.")
        return

    # Filtra le colonne di interesse
    player_data = player_data[features + ['defending']]

    # Crea le sequenze temporali
    player_data_values = player_data.values
    X_player = []
    for i in range(len(player_data_values) - timesteps):
        X_player.append(player_data_values[i:i + timesteps, :-1])  # Input: colonne utili eccetto il target

    X_player = np.array(X_player)

    # Normalizza i dati
    X_player_scaled = scaler_X.transform(X_player.reshape(-1, len(features)))
    X_player_scaled = X_player_scaled.reshape(X_player.shape)

    # Effettua la previsione
    y_pred_scaled = model.predict(X_player_scaled)

    # Inverti la normalizzazione per ottenere il valore originale di "defending"
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

    # Stampa il risultato
    print(f"La previsione per 'defending' del giocatore {player_name} è: {y_pred_original[-1][0]:.4f}")


# Esempio: previsioni per un giocatore
player_name = "Matteo Ricci"  # Modifica il nome del giocatore qui
predict_defending(player_name)
