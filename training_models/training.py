import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.optimizers import Adam

from training_models.functions import *

#CARICAMENTO E PREPRAZIONE DEL DATASET
df = load_dataset("../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv")
df = df.sort_values(by=['long_name', 'fifa_version'])

#DEFINIZIONE DELLE FEATURE E DEI TARGET
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


#CREAZIONE DELLE SEQUENZE TEMPORALI PER OGNI TARGET
timesteps = 5  #Lunghezza della sequenza temporale
#I DIZIONARI X_DICT E Y_DICT MEMORIZZANO LE SEQUENZE PER OGNI TARGET
X_dict = {}
y_dict = {}
for target_name in target:
    X, y = [], []
    feature_columns = features[target_name]

    #RAGGRUPPIAMO PER GIOCATORE
    for player, player_data in df.groupby('long_name'):
        if len(player_data) >= timesteps + 1:  #Deve avere abbastanza versioni
            player_data_values = player_data[feature_columns + [target_name]].values
            for i in range(len(player_data_values) - timesteps):
                X.append(player_data_values[i:i + timesteps, :-1])  #Input: colonne utili eccetto il target
                y.append(
                    player_data_values[i + timesteps, -1])  #Target: valore del target (es. 'defending') successivo

    #CONVERTIAMO IN ARRAY NUMPHY
    X_dict[target_name] = np.array(X)  #Shape: (num_samples, timesteps, num_features)
    y_dict[target_name] = np.array(y)  #Shape: (num_samples,)

    print(f"Forma di X per {target_name}: {X_dict[target_name].shape}")
    print(f"Forma di y per {target_name}: {y_dict[target_name].shape}")


#COSTRUZIONE DEL MODELLO LSTM
#CREIAMO UN MODELLO LSTM CON 3 STRATI LSTM, OGNI STRATO E1 SEGUITO DA UN DROPOUT E DUE STRATI DENSE.
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
        Dense(1)  #PREVEDIAMO UN SINGOLO VALORE PER OGNI TARGET
    ])
    #COMPILIAMO CON L'OTTIMIZZATORE ADAM E UNA FUNZIONE DI PERDITA PERSONALIZZATA, CON LA METRICA MAE
    model.compile(optimizer=Adam(learning_rate=0.001), loss=weighted_loss, metrics=['mae'])
    return model


#ADDESTRAMENTO E VALUTAZIONE DEL MODELLO PER OGNI TARGET
for target_name in target:
    print(f"\nInizio addestramento per il target: {target_name}")

    #COSTRUIAMO IL MODELLO
    model = build_model(input_shape=(timesteps, len(features[target_name])))

    #PER EVITARE OVERFITTING IMPOSTIAMO L'EARLY STOPPING
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model_path = f"../models/{target_name}/{target_name}_model.keras"

    #SALVIAMO IL MODELLO
    model_checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)

    #NORMALIZZAZIONE SPECIFICA PER OGNI TARGET
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    #Appiattire X per scalare (timesteps non sono considerati durante la normalizzazione)
    X_flat = X_dict[target_name].reshape(-1, len(features[target_name]))
    X_flat_scaled = scaler_X.fit_transform(X_flat)
    X_dict[target_name] = X_flat_scaled.reshape(X_dict[target_name].shape)

    #Ridimensiona y
    y_dict[target_name] = scaler_y.fit_transform(y_dict[target_name].reshape(-1, 1))

    #SUDDIVIAMO I DATI IN TRAINING E TEST, ADDESTRAMENTO 80% E TEST 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X_dict[target_name], y_dict[target_name], test_size=0.2, random_state=42
    )

    #Addestramento del modello
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_split=0.2, verbose=1,
                        callbacks=[early_stopping, model_checkpoint])

    #SALVIAMO IL MODELLO CON LA FUNZIONE DI PERDITA PERSONALIZZATA
    model.save(model_path)

    #SALVATAGGIO DEL MODELLO E DEGLI SCALER
    scaler_X_path = f"../models/{target_name}/scaler_X.pkl"
    scaler_y_path = f"../models/{target_name}/scaler_y.pkl"
    with open(scaler_X_path, 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)

    #VALUTIAMO IL MODELLO
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Errore assoluto medio (MAE) per {target_name}: {mae}")

    #Predizioni per visualizzazione
    y_pred_scaled = model.predict(X_test)

    #Invertire la normalizzazione per interpretare i risultati
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

    #GRAFICO DELLA LOSS
    plt.plot(history.history['loss'], label=f'Training Loss ({target_name})')
    plt.plot(history.history['val_loss'], label=f'Validation Loss ({target_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{target_name} - Loss')
    plt.legend()
    plt.show()

    #VISUALIZZIAMO LE PREDIZIONI VS VALORI REALI
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_original[:100], label='Valori reali', color='blue')
    plt.plot(y_pred_original[:100], label='Predizioni del modello', color='red')
    plt.xlabel('Campioni')
    plt.ylabel(f'{target_name}')
    plt.title(f'Predizioni vs Realità per {target_name}')
    plt.legend()
    plt.show()