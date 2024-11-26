import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

# Carica il dataset
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv')

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
    LSTM(64, input_shape=(timesteps, len(features)), activation='tanh', return_sequences=True),
    Dropout(0.2),  # Aggiungi dropout per evitare overfitting
    LSTM(64, activation='tanh'),
    Dropout(0.2),  # Aggiungi un altro layer di dropout
    Dense(1)  # Previsione di un singolo valore (es. 'defending')
])

# Ottimizzatori da provare
optimizers = {
    'Adam': Adam(),
    'RMSprop': RMSprop(),
}

# EarlyStopping per evitare overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Testiamo vari ottimizzatori
for optimizer_name, optimizer in optimizers.items():
    print(f"\nAllenamento con {optimizer_name}...\n")

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Addestramento del modello
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1,
                        callbacks=[early_stopping])

    # Valutazione del modello
    loss, mae = model.evaluate(X_test, y_test, verbose=1)

    print(f"Errore assoluto medio (MAE) con {optimizer_name}: {mae}")

    # Predizioni per visualizzazione
    y_pred_scaled = model.predict(X_test)

    # Invertire la normalizzazione per interpretare i risultati
    y_test_original = scaler_y.inverse_transform(y_test)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

    # Grafico della Loss
    plt.plot(history.history['loss'], label=f'{optimizer_name} - Training Loss')
    plt.plot(history.history['val_loss'], label=f'{optimizer_name} - Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{optimizer_name} - Loss')
    plt.legend()
    plt.show()

    # Grafico del MAE
    plt.plot(history.history['mae'], label=f'{optimizer_name} - Training MAE')
    plt.plot(history.history['val_mae'], label=f'{optimizer_name} - Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title(f'{optimizer_name} - Mean Absolute Error')
    plt.legend()
    plt.show()

    # Visualizzare le predizioni vs. valori reali
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_original[:100], label='Valori reali (defending)', color='blue')
    plt.plot(y_pred_original[:100], label='Predizioni del modello', color='red')
    plt.xlabel('Campioni')
    plt.ylabel('Defending')
    plt.title(f'Predizioni vs Valori Reali ({optimizer_name} - First 100 samples)')
    plt.legend()
    plt.show()

