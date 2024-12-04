from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import pandas as pd

df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv')

# Aggiunta di feature extra
df['experience'] = df.groupby('long_name').cumcount() + 1
df['age_trend'] = df['age'].diff().fillna(0)

# Carica i parametri
with open('../models/model_params.pkl', 'rb') as f:
    params = pickle.load(f)

# Estrai i parametri
features = params['features']
timesteps = params['timesteps']
num_folds = 5  # Numero di fold per la k-fold validation

# Prepara i dati
X, y = [], []
for player, player_data in df.groupby('long_name'):
    if len(player_data) >= timesteps + 1:
        player_data = player_data[features + ['defending']].values
        for i in range(len(player_data) - timesteps):
            X.append(player_data[i:i + timesteps, :-1])
            y.append(player_data[i + timesteps, -1])

X = np.array(X)
y = np.array(y)

# Normalizza i dati
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_flat = X.reshape(-1, len(features))
X_scaled = scaler_X.fit_transform(X_flat).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=num_folds)
metrics = []

for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

    # Definisci il modello LSTM
    model = keras.Sequential([
        LSTM(128, input_shape=(timesteps, len(features)), activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Addestramento
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    # Valutazione
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    metrics.append({'loss': loss, 'mae': mae})

# Media e deviazione standard delle metriche
average_metrics = {
    'loss': np.mean([m['loss'] for m in metrics]),
    'mae': np.mean([m['mae'] for m in metrics]),
    'std_loss': np.std([m['loss'] for m in metrics]),
    'std_mae': np.std([m['mae'] for m in metrics]),
}

print(f"Risultati medi:\nLoss: {average_metrics['loss']:.4f} ± {average_metrics['std_loss']:.4f}")
print(f"MAE: {average_metrics['mae']:.4f} ± {average_metrics['std_mae']:.4f}")
