#Il codice carica un dataset di giocatori di FIFA, normalizza i dati utilizzando
#degli scaler pre-addestrati, prepara sequenze temporali per l'input del modello,
#divide i dati in set di addestramento e di test, e infine calcola alcune statistiche
#descrittive (media e deviazione standard) per ciascuna delle caratteristiche del set di dati.

import numpy as np #OPERAZIONI MATEMATICHE E MANIPOLAZIONE DEGLI ARRAY
from sklearn.model_selection import train_test_split #SUDDIVIDERE I DATI IN SET DI ADDESTRAMENTO E TEST
import pickle #SERIALIZZAZIONE E DESERALIZZAZIONE DI OGGETTI PY
from training_models.functions import load_dataset #FUNZIONE DEFINITA CHE CARICA IL DATASET PRE-PROCESSATO


#CARICAMENTO DEL DATASET
df = load_dataset('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv')

#CARICAMENTO DI SCALERX E SCALERY. USATI PER NORMALIZZARE I DATI IN MODO COERENTE CON IL PRECEDENTE MODELLO
with open('../models/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('../models/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)


#VENGONO CARICATI I PARAMETRI
with open('../models/model_params.pkl', 'rb') as f:
    params = pickle.load(f)

#I PARAMETRI ESTRATTI SONO FEATURES E IL NUMERO DI TIMESTEPS
features = params['features']
timesteps = params['timesteps']

#PREPRAZIONE DEI DATI PER IL MODELLO DI APPRENDIMENTO
X, y = [], []
for player, player_data in df.groupby('long_name'):
    if len(player_data) >= timesteps + 1:
        #VIENE SELEZIONATO UN SOTTOINSIEME DEI DATI DEL GIOCATORE,
        #CHE CONTIENE SOLO LE COLONNE SPECIFICATE IN FEATURES E LA COLONNA 'DEFENDING'
        player_data = player_data[features + ['defending']].values
        #I DATI VENGONO ORGANIZZATI IN SEQUENZE TEMPORALI
        #PER OGNI SEQUENZA TEMPORALE DI LUNGHEZZA TIMESTEPS, VIENE CREATO
        #UN ARRAY X DI CARATTERISTICHE E UN ARRAY Y CHE CONTIENE IL TARGET
        for i in range(len(player_data) - timesteps):
            X.append(player_data[i:i + timesteps, :-1])
            y.append(player_data[i + timesteps, -1])

#GLI ARRAY X E Y VENGONO CONVERTITI IN ARRAY NUMPY PER FACILITARNE LA MANIPOLAZIONE ED ELABORAZIONE
X = np.array(X)
y = np.array(y)

#NORMALIZZIAMO I DATI
#X VIENE APPIATTITO IN UN ARRAY BIDIMENSIONALE PER ADATTARSI ALL'INPUT RICHIESTO DALLO SCALER
X_flat = X.reshape(-1, len(features))
X_flat_scaled = scaler_X.transform(X_flat)
X_scaled = X_flat_scaled.reshape(X.shape)

y_scaled = scaler_y.transform(y.reshape(-1, 1))

#DIVIDIAMO IN SET DI ADDESTRAMENTO E DI TEST
#IL 30% DEI DATI VIENE USATO PER IL TEST
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

#CALCOLO DELLE STATISTICHE DESCRITTIVE PER LE CARATTERISTICHE
for feature in features:

    #PER OGNI CARATTERISTICA NEL DATASET:
    #VENGONO ESTRATTI I VALORI CORRISPONDENTI DALLA PARTE DI ADDESTRAMENTO E DI TEST
    train_values = X_train[:, :, features.index(feature)].flatten()
    test_values = X_test[:, :, features.index(feature)].flatten()

    print(f"Feature: {feature}")
    print(f"Training Set - Media: {train_values.mean():.4f}, Deviazione Standard: {train_values.std():.4f}")
    print(f"Test Set - Media: {test_values.mean():.4f}, Deviazione Standard: {test_values.std():.4f}")
    print("-" * 40)
