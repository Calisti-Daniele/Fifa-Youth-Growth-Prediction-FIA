import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from training_models.functions import *

# Carica il dataset
df = load_dataset("../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv")

# Ordina per giocatore e fifa_version
df = df.sort_values(by=['long_name', 'fifa_version'])

# Definisci le feature di input e il target
features = {
    'overall': ['potential', 'passing', 'dribbling', 'movement_reactions', 'mentality_composure'],
    'potential': ['overall', 'passing', 'dribbling'],
    'shooting': ['passing', 'dribbling', 'attacking_finishing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
                 'skill_long_passing', 'skill_ball_control', 'movement_agility', 'power_shot_power', 'power_long_shots',
                 'mentality_positioning', 'mentality_vision', 'mentality_penalties'],
    'passing': ['shooting', 'dribbling', 'attacking_crossing', 'attacking_short_passing', 'skill_dribbling', 'skill_curve',
                'skill_long_passing', 'skill_fk_accuracy', 'skill_ball_control', 'power_long_shots', 'mentality_vision',
                'mentality_positioning'],
    'dribbling': ['shooting', 'passing', 'attacking_crossing', 'attacking_finishing', 'attacking_volleys',
                  'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_ball_control', 'movement_acceleration',
                  'movement_agility', 'movement_balance', 'power_long_shots', 'mentality_positioning',
                  'mentality_vision'],
    'defending': ['physic', 'mentality_aggression', 'mentality_interceptions', 'defending_marking_awareness',
                  'defending_standing_tackle', 'defending_sliding_tackle'],
    'physic': ['defending', 'power_strength', 'mentality_aggression', 'mentality_interceptions']
}

target = list(features.keys())

# Parametri per LSTM
timesteps = 5  # Lunghezza della sequenza temporale

# Definisci X e y come dizionari separati per ogni target
X_dict = {}
y_dict = {}

for target_name in target:
    X, y = [], []
    feature_columns = features[target_name]

    # Raggruppa per giocatore e crea le sequenze temporali
    for player, player_data in df.groupby('long_name'):
        if len(player_data) >= timesteps + 1:  # Deve avere abbastanza versioni
            player_data_values = player_data[feature_columns + [target_name]].values
            for i in range(len(player_data_values) - timesteps):
                X.append(player_data_values[i:i + timesteps, :-1])  # Input: colonne utili eccetto il target
                y.append(
                    player_data_values[i + timesteps, -1])  # Target: valore del target (es. 'defending') successivo

    # Converti in array numpy
    X_dict[target_name] = np.array(X)  # Shape: (num_samples, timesteps, num_features)
    y_dict[target_name] = np.array(y)  # Shape: (num_samples,)

    print(f"Forma di X per {target_name}: {X_dict[target_name].shape}")
    print(f"Forma di y per {target_name}: {y_dict[target_name].shape}")


# Definisci il modello LSTM con Dropout
def build_model(input_shape):
    model = keras.Sequential([
        LSTM(128, input_shape=input_shape, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(128, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Previsione di un singolo valore per ogni target
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=weighted_loss, metrics=['mae'])
    return model


# Addestramento e valutazione del modello per ogni target
for target_name in target:
    print(f"\nInizio addestramento per il target: {target_name}")

    # Costruisci il modello
    model = build_model(input_shape=(timesteps, len(features[target_name])))

    # EarlyStopping per evitare overfitting
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model_path = f"../models/{target_name}/{target_name}_model.keras"

    # Salviamo il modello
    model_checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)

    # Normalizzazione specifica per ogni target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Appiattire X per scalare (timesteps non sono considerati durante la normalizzazione)
    X_flat = X_dict[target_name].reshape(-1, len(features[target_name]))
    X_flat_scaled = scaler_X.fit_transform(X_flat)
    X_dict[target_name] = X_flat_scaled.reshape(X_dict[target_name].shape)

    # Ridimensiona y
    y_dict[target_name] = scaler_y.fit_transform(y_dict[target_name].reshape(-1, 1))

    # Divisione in train/test per ogni target
    X_train, X_test, y_train, y_test = train_test_split(
        X_dict[target_name], y_dict[target_name], test_size=0.2, random_state=42
    )

    # Addestramento del modello
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_split=0.2, verbose=1,
                        callbacks=[early_stopping, model_checkpoint])

    # Salva il modello con la funzione personalizzata
    model.save(model_path)

    # Salva gli scaler e parametri
    scaler_X_path = f"../models/{target_name}/scaler_X.pkl"
    scaler_y_path = f"../models/{target_name}/scaler_y.pkl"
    with open(scaler_X_path, 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)

    # Valutazione del modello
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Errore assoluto medio (MAE) per {target_name}: {mae}")

    # Predizioni per visualizzazione
    y_pred_scaled = model.predict(X_test)

    # Invertire la normalizzazione per interpretare i risultati
    y_test_original = scaler_y.inverse_transform(y_test)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    print(f"\nPerformance del modello per {target_name}:")
    print(f"- MAE: {mae:.4f}")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- R2-Score: {r2:.4f}")

    # Grafico della Loss
    plt.plot(history.history['loss'], label=f'Training Loss ({target_name})')
    plt.plot(history.history['val_loss'], label=f'Validation Loss ({target_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{target_name} - Loss')
    plt.legend()
    plt.show()

    # Visualizzare le predizioni vs. valori reali
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_original[:100], label='Valori reali', color='blue')
    plt.plot(y_pred_original[:100], label='Predizioni del modello', color='red')
    plt.xlabel('Campioni')
    plt.ylabel(f'{target_name}')
    plt.title(f'Predizioni vs Realità per {target_name}')
    plt.legend()
    plt.show()
