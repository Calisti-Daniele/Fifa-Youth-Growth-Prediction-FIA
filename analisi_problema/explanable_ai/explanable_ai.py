import shap
import shap.maskers as msk
import numpy as np
import keras
import pickle
from training.functions import weighted_loss, load_dataset

# Carica il modello e i parametri salvati
model_path = '../../models/fia_model.keras'
scaler_X_path = '../../models/scaler_X.pkl'
params_path = '../../models/model_params.pkl'

# Carica il modello
model = keras.models.load_model(model_path, custom_objects={"weighted_loss": weighted_loss})

# Carica gli scaler
with open(scaler_X_path, 'rb') as f:
    scaler_X = pickle.load(f)

# Carica i parametri
with open(params_path, 'rb') as f:
    params = pickle.load(f)

# Estrai le feature
features = params['features']

# Carica il dataset
dataset_path = '../../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv'
df = load_dataset(dataset_path)

# Prepara i dati per l'analisi
timesteps = params['timesteps']
X = []
for player, player_data in df.groupby('long_name'):
    if len(player_data) >= timesteps:
        player_data = player_data[features].values
        X.append(player_data[-timesteps:, :])  # Prendi le ultime `timesteps`

X = np.array(X)  # Shape: (num_players, timesteps, num_features)

# Normalizza i dati
X_flat = X.reshape(-1, len(features))
X_scaled = scaler_X.transform(X_flat).reshape(X.shape)

# Usa GradientExplainer
explainer = shap.GradientExplainer(model, X_scaled)

# Calcola i valori SHAP per un sottoinsieme dei dati
shap_values = explainer.shap_values(X_scaled[:100])  # Limita a 100 campioni per prestazioni migliori

# Aggrega i valori SHAP lungo i timesteps
shap_values_mean = np.mean(shap_values[0], axis=1)  # Media lungo i timesteps

# Visualizza l'importanza globale delle feature
shap.summary_plot(shap_values_mean, X_scaled[:100].mean(axis=1), feature_names=features)

# Visualizza l'impatto delle feature su una singola predizione
baseline_prediction = np.mean(model.predict(X_scaled[:100]))
shap.force_plot(baseline_prediction, shap_values[0], features=features)