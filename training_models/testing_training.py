"""
In questo file carichiamo un dataset di giocatori di calcio, prepariamo i dati creando sequenze temporali,
normalizziamo i dati, definiamo e addestriamo un modello LSTM per prevedere le abilità difensive di un giocatore,
e salviamo il modello insieme agli scaler per future predizioni.
Infine, il modello viene valutato e calcolato l'errore di previsione.

"""


import keras #costruzione e addestramento di reti neurali
import numpy as np #manipolazione dei dati
import pandas as pd #manipolazione dei dati
import pickle #per salvare il modello addestrato e gli scaler
from sklearn.metrics import mean_absolute_error #per la suddivisione del dataset in training e test e per la normalizzazione dei dati
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.api.layers import LSTM, Dense, Dropout #per la gestione dei layer e dell'ottimizzazione della rete neurale
from keras.api.regularizers import l2
from keras.api.optimizers import Adam

from training_models.functions import weighted_loss

#CARICAMENTO DEL DATASET
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv')
model_path = '../models/fia_model.keras'
df = df.sort_values(by=['long_name', 'fifa_version'])

#AGGIUNTA DI FEATURE EXTRA
#EXPECERIENCE: CALCOLA IL NUMERO DI STAGIONI DI OGNI GIOCATORE, CONTANDO I RECORD PER GIOCATORE
df['experience'] = df.groupby('long_name').cumcount() + 1

#AGE_TREND: CALCOLA LA VARIAZIONE DI ETA' DI OGNI GIOCARE RISPETTO ALLA STAGIONE PRECEDENTE
df['age_trend'] = df['age'].diff().fillna(0)

#DEFINIAMO:
#FEATURES: LE VARIABILI CHE VERRANNO USATE COME INPUT PER IL MODELLO
features = [
    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
    'mentality_interceptions', 'mentality_aggression', 'physic', 'experience', 'age_trend'
]
#TARGET: L'ABILITA' DIFENSIVA CHE IL MODELLO DEVE PREVEDERE
target = 'defending'

#RAGGRUPA IL DATASET PER CIASCUN GIOCATORE
#CREIAMO SEQUENZE TEMPORALI DI 5 STAGIONI
timesteps = 5
#X CONTIENE LE SEQUENZE TEMPORALI DI INPUT, MENTRE Y CONTIENE I TARGET RELATIVI AL PERIODO SUCCESSIVO
X, y = [], []
for player, player_data in df.groupby('long_name'):
    if len(player_data) >= timesteps + 1:
        player_data = player_data[features + [target]].values
        for i in range(len(player_data) - timesteps):
            X.append(player_data[i:i + timesteps, :-1])
            y.append(player_data[i + timesteps, -1])

X = np.array(X)
y = np.array(y)

print("Forma di X (Input):", X.shape)
print("Forma di y (Target):", y.shape)



#NORMALIZZIAMO LE FEATURE DI INPUT E I TARGET IN UN INTERVALLO DA 0 A 1 UTILIZZANDO MINMAXSCALER DI SCIKIT-LEARN
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_flat = X.reshape(-1, len(features))
X_scaled = scaler_X.fit_transform(X_flat).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

#SUDDIVISIONE DEL DATASET IN TRAINING E TEST
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

#DEFINIZIONE DEL MODELLO LSTM
model = keras.Sequential([
    LSTM(128, input_shape=(timesteps, len(features)), activation='tanh', return_sequences=True,
         kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    LSTM(64, activation='tanh', return_sequences=False, kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])

#COMPILIAMO IL MODELLO UTILIZZANDO L'OTTIMIZZATORE ADAM PER MINIMIZZARE LA FUNZIONE DI PERDITA
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=weighted_loss, metrics=['mae'])

#ADDESTRAMENTO DEL MODELLO
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint])

#SALVATAGGIO DEL MODELLO E DEGLI SCALER
model.save(model_path)
with open('../models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('../models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

params = {"features": features, "timesteps": timesteps}
with open('../models/model_params.pkl', 'wb') as f:
    pickle.dump(params, f)

#VALUTAZIONE DEL MODELLO
loss, mae = model.evaluate(X_test, y_test, verbose=1)
y_pred = scaler_y.inverse_transform(model.predict(X_test))
y_test_original = scaler_y.inverse_transform(y_test)
print(f"MAE: {mean_absolute_error(y_test_original, y_pred):.2f}")
