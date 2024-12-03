import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Carica il dataset
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv')
model_path = '../models/fia_model.keras'
df = df.sort_values(by=['long_name', 'fifa_version'])

# Aggiunta di feature extra
df['experience'] = df.groupby('long_name').cumcount() + 1
df['age_trend'] = df['age'].diff().fillna(0)

# Nuove feature
features = [
    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
    'mentality_interceptions', 'mentality_aggression', 'physic', 'experience', 'age_trend'
]
target = 'defending'

timesteps = 5
X, y = [], []

# Raggruppa per giocatore e crea le sequenze temporali
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

# Normalizzazione
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_flat = X.reshape(-1, len(features))
X_scaled = scaler_X.fit_transform(X_flat).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Divisione in train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)


@keras.utils.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    weights = 1 / (1 + tf.abs(y_true - y_pred))  # Più lontano, più peso
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))


# Modello LSTM con regolarizzazione
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

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=weighted_loss, metrics=['mae'])

# Addestramento
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint])

# Salva il modello con la funzione personalizzata
model.save(model_path)

# Salva gli scaler e parametri
with open('../models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('../models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

params = {"features": features, "timesteps": timesteps}
with open('../models/model_params.pkl', 'wb') as f:
    pickle.dump(params, f)

# Valutazione
loss, mae = model.evaluate(X_test, y_test, verbose=1)
y_pred = scaler_y.inverse_transform(model.predict(X_test))
y_test_original = scaler_y.inverse_transform(y_test)

print(f"MAE: {mean_absolute_error(y_test_original, y_pred):.2f}")
