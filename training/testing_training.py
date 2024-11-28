import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Carica il dataset
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv')

model_path = '../models/fia_model.keras'
# Ordina per giocatore e fifa_version
df = df.sort_values(by=['short_name', 'fifa_version'])

# Definisci le feature di input e il target
features = [
    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
    'mentality_interceptions', 'mentality_aggression', 'physic'
]
target = 'defending'

# Parametri per LSTM
timesteps = 5  # Lunghezza della sequenza temporale
X, y = [], []

# Raggruppa per giocatore e crea le sequenze temporali
for player, player_data in df.groupby('short_name'):
    if len(player_data) >= timesteps + 1:  # Deve avere abbastanza versioni
        player_data = player_data[features + [target]].values
        for i in range(len(player_data) - timesteps):
            X.append(player_data[i:i + timesteps, :-1])  # Input: colonne utili eccetto il target
            y.append(player_data[i + timesteps, -1])  # Target: valore di "defending" successivo

# Converti in array numpy
X = np.array(X)  # Shape: (num_samples, timesteps, num_features)
y = np.array(y)  # Shape: (num_samples,)

print("Forma di X (Input):", X.shape)
print("Forma di y (Target):", y.shape)

# Normalizzazione delle feature
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Appiattire X per scalare (timesteps non sono considerati durante la normalizzazione)
X_flat = X.reshape(-1, len(features))
X_flat_scaled = scaler_X.fit_transform(X_flat)
X_scaled = X_flat_scaled.reshape(X.shape)  # Riformare nella shape originale

# Ridimensiona y
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Divisione in train e test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Definisci il modello LSTM con Dropout
model = keras.Sequential([
    LSTM(128, input_shape=(timesteps, len(features)), activation='tanh', return_sequences=True),
    Dropout(0.3),
    LSTM(128, activation='tanh', return_sequences=True),
    Dropout(0.3),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Previsione di un singolo valore (es. 'defending')
])

# Ottimizzatore
optimizer = Adam()

# EarlyStopping per evitare overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Salviamo il modello
model_checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)

# Compilazione del modello
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Addestramento del modello
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1,
                    callbacks=[early_stopping, model_checkpoint])

if os.path.exists(model_path):
    print(f"Modello salvato correttamente in: {model_path}")
else:
    print(f"Modello non salvato correttamente in: {model_path}")

# Valutazione del modello
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Errore assoluto medio (MAE) con Adam: {mae}")

# Predizioni per visualizzazione
y_pred_scaled = model.predict(X_test)

# Invertire la normalizzazione per interpretare i risultati
y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"\nPerformance del modello caricato:")
print(f"- MAE: {mae:.4f}")
print(f"- MSE: {mse:.4f}")
print(f"- RMSE: {rmse:.4f}")
print(f"- R2-Score: {r2:.4f}")

# Grafico della Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Adam - Loss')
plt.legend()
plt.show()

# Grafico del MAE
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Adam - Mean Absolute Error')
plt.legend()
plt.show()

# Visualizzare le predizioni vs. valori reali
plt.figure(figsize=(10, 5))
plt.plot(y_test_original[:100], label='Valori reali (defending)', color='blue')
plt.plot(y_pred_original[:100], label='Predizioni del modello', color='red')
plt.xlabel('Campioni')
plt.ylabel('Defending')
plt.title('Predizioni vs Valori Reali (Adam - First 100 samples)')
plt.legend()
plt.show()
