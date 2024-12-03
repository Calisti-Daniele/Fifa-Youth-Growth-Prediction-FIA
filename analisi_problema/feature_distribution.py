import matplotlib.pyplot as plt
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

# Carica il dataset
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv')

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
X_flat = X.reshape(-1, len(features))
X_flat_scaled = scaler_X.transform(X_flat)
X_scaled = X_flat_scaled.reshape(X.shape)

y_scaled = scaler_y.transform(y.reshape(-1, 1))

# Divisione in train e test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

for feature in features:
    train_values = X_train[:, :, features.index(feature)].flatten()
    test_values = X_test[:, :, features.index(feature)].flatten()

    print(f"Feature: {feature}")
    print(f"Training Set - Media: {train_values.mean():.4f}, Deviazione Standard: {train_values.std():.4f}")
    print(f"Test Set - Media: {test_values.mean():.4f}, Deviazione Standard: {test_values.std():.4f}")
    print("-" * 40)
