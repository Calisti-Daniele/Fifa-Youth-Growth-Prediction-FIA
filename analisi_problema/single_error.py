#Il codice carica i dati storici dei giocatori (2015-2023), il modello addestrato e gli scaler.
#Prepara i dati di input usando una finestra temporale di lunghezza definita (timesteps).
#Effettua le predizioni per ogni giocatore.
#Confronta le predizioni con i valori effettivi nel dataset 2024.
#Salva i risultati in un file CSV, ordinando per la differenza tra predizioni e valori reali.

import pickle #CARICARE SCALER E PARAMETRI SALVATI
import numpy as np #GESTIRE DATI NUMERICI
import pandas as pd #GESTIRE I DATAFRAME
import keras #PER CARICARE IL MODELLO DI DEEP LEARNING

from training_models.functions import load_dataset
from training_models.functions import weighted_loss

#PERCORSI AI FILE SALVATI
dataset_path = '../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv'
dataset_fc_24_path = '../datasets/dataset_fc_24.csv'
model_path = '../models/fia_model.keras'
scaler_X_path = '../models/scaler_X.pkl'
scaler_y_path = '../models/scaler_y.pkl'
params_path = '../models/model_params.pkl'

#CARICHIAMO IL DATASET FIFA 15-23
df = load_dataset(dataset_path)

#LO ORDINIAMO PER NOME DEL GIOCATORE E PER LA VERSIONE DEL FIFA
df = df.sort_values(by=['long_name', 'fifa_version'])


#CARICA IL MODELLO SALVATO: UNA RETE NEURALE ADDESTRATA
model = keras.models.load_model(model_path, custom_objects={"weighted_loss": weighted_loss})

#CARICHIAMO GLI SCALER
with open(scaler_X_path, 'rb') as f:
    scaler_X = pickle.load(f)
with open(scaler_y_path, 'rb') as f:
    scaler_y = pickle.load(f)

#CARICHIAMO I PARAMETRI
with open(params_path, 'rb') as f:
    params = pickle.load(f)

#ESTRAIAMO I PARAMETRI:

#FEATURES: LISTA DELLE COLONNE DEL DATASET DA UTILIZZARE PER LE PREDIZIONI
features = params['features']

#TIMESTEPS: LUNGHEZZA DELLA SEQUENZA TEMPORALE
timesteps = params['timesteps']

#PREPARIAMO I DATI PER LA PREVISIONE
X, player_names = [], []
for player, player_data in df.groupby('long_name'):
    if len(player_data) >= timesteps:
        player_data = player_data[features].values
        X.append(player_data[-timesteps:, :])  # Prendi le ultime `timesteps`
        player_names.append(player)  # Salva il nome del giocatore

X = np.array(X)  # Shape: (num_players, timesteps, num_features)

#NORMALIZZIAMO I DATI
X_flat = X.reshape(-1, len(features))
X_scaled = scaler_X.transform(X_flat).reshape(X.shape)

#PREDIZIONI
y_pred_scaled = model.predict(X_scaled)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

#CARICHIAMO IL DATASET DI FC 24
df_fc_24 = pd.read_csv(dataset_fc_24_path)

#AGGIUNGIAMO:
#EXPERIENCE: INDICA L'ESPERIENZA DEL GIOCATOR, CONTA QUANTE VOLTE UN GIOCATORE APPARE NEL DATASET
df_fc_24['experience'] = df_fc_24.groupby('long_name').cumcount() + 1
#AGE_TREND: CALCOLA LA DIFFERENZA DI ETÀ TRA I VARI RECORD DELLO STESSO GIOCATORE, PER OSSERVARE
#COME CAMBIA L'ETÀ TRA I DIVERSI ANNI
df_fc_24['age_trend'] = df_fc_24.groupby('long_name')['Age'].diff().fillna(0)

#CONFRONTIAMO LE PREDIZIONI CON I VALORI REALI
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

#CREIAMO UN DATAFRAME PER SALVARLO
comparison_df = pd.DataFrame(results)

#ORDINIAMO PER DIFFERENZA DESCRESCENTE
comparison_df = comparison_df.sort_values(by='Differenza', ascending=False)

#MOSTRIAMO I PRIMI 10 RISULTATI
print(comparison_df.head(10))

#SALVIAMO IL DATAFRAME PER ULTERIORI ANALISI
comparison_df.to_csv('outputs/fia_comparison_2024.csv', index=False)
print("Confronto salvato in 'outputs/fia_comparison_2024.csv'")
