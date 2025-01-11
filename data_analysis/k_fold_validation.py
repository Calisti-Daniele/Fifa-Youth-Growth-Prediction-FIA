"""
Questo script esegue la valutazione di modelli predittivi per più target utilizzando una validazione temporale
(TimeSeriesSplit). L'obiettivo è analizzare le prestazioni dei modelli per prevedere caratteristiche di giocatori
di calcio (ad esempio: overall, shooting, passing, ecc.) basandosi su dati FIFA.

Passaggi principali:
1. Caricamento e preprocessamento del dataset:
   - Il dataset viene ordinato per giocatore e versione FIFA per preservare l'ordine temporale.
   - Vengono aggiunte nuove feature (`experience` e `age_trend`) per arricchire l'analisi.

2. Definizione dei target e delle feature:
   - Ogni target è associato a un set specifico di feature che saranno utilizzate come input per il modello.

3. Valutazione con TimeSeriesSplit:
   - I dati sono trasformati in sequenze temporali di lunghezza definita (`timesteps`).
   - Viene effettuata la normalizzazione dei dati utilizzando scaler pre-addestrati.
   - Ogni modello è valutato su più fold temporali per calcolare metriche di prestazione (Loss, MAE).

4. Riepilogo dei risultati:
   - I risultati (media e deviazione standard di Loss e MAE) sono riportati per ciascun target.

Utilizzo:
Questo script è utile per confrontare le prestazioni di modelli predittivi in contesti temporali e analizzare
le relazioni tra le feature e i target nei dati FIFA.

Prerequisiti:
- Modelli salvati (formato `.keras`) e scaler (`scaler_X.pkl`, `scaler_y.pkl`) devono essere disponibili nella directory dei modelli.
- Dataset preprocessato e non normalizzato deve essere accessibile al percorso specificato.

Output:
- Metriche aggregate per ogni target (Loss, MAE) calcolate su più fold temporali.
"""


from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pickle
import os
from training_models.functions import *


#PERCORSO DEI FILE
models_dir = '../models/'
dataset_path = '../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv'
output_dir = 'outputs/'

#CARICAMENTO DEL DATASET
print("Caricamento del dataset...")
df = load_dataset(dataset_path)
df = df.sort_values(by=['long_name', 'fifa_version'])

#AGGIUNGIAMO:
#EXPERIENCE: INDICA L'ESPERIENZA DEL GIOCATORE, CONTA QUANTE VOLTE UN GIOCATORE APPARE NEL DATASET
df['experience'] = df.groupby('long_name').cumcount() + 1
#AGE_TREND: CALCOLA LA DIFFERENZA DI ETÀ TRA I VARI RECORD DELLO STESSO GIOCATORE, PER OSSERVARE
#COME CAMBIA L'ETÀ TRA I DIVERSI ANNI
df['age_trend'] = df.groupby('long_name')['age'].diff().fillna(0)

#DEFINIZIONE DEI TARGET E DELLE FEATURES
features_dict = {
    'overall': ['potential', 'passing', 'dribbling', 'movement_reactions', 'mentality_composure'],
    'potential': ['overall', 'passing', 'dribbling'],
    'shooting': ['passing', 'dribbling', 'attacking_finishing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
                 'skill_long_passing', 'skill_ball_control', 'movement_agility', 'power_shot_power', 'power_long_shots',
                 'mentality_positioning', 'mentality_vision', 'mentality_penalties'],
    'passing': ['shooting', 'dribbling', 'attacking_crossing', 'attacking_short_passing', 'skill_dribbling',
                'skill_curve',
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

timesteps = 5
num_folds = 5


#Funzione principale per valutare ogni target con TimeSeriesSplit
def evaluate_model_with_tscv(target_name, features, model_path, scaler_X_path, scaler_y_path):
    print(f"\nInizio valutazione per il target: {target_name}")

    #CARICAMENTO DEL MODELLO E DEGLI SCALER
    model = keras.models.load_model(model_path)
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    #PREPARAZIONE DEI DATI
    X, y = [], []
    for player, player_data in df.groupby('long_name'):
        if len(player_data) >= timesteps + 1:
            player_data_values = player_data[features + [target_name]].values
            for i in range(len(player_data_values) - timesteps):
                X.append(player_data_values[i:i + timesteps, :-1])  # Input
                y.append(player_data_values[i + timesteps, -1])  # Target

    X = np.array(X)
    y = np.array(y)

    #NORMALIZZIAMO I DATI
    X_flat = X.reshape(-1, len(features))
    X_scaled = scaler_X.transform(X_flat).reshape(X.shape)
    y_scaled = scaler_y.transform(y.reshape(-1, 1))

    #TIMESERIESSPLIT
    tscv = TimeSeriesSplit(n_splits=num_folds)
    metrics = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):
        print(f"\nFold {fold + 1}/{num_folds}")
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_scaled[train_index], y_scaled[test_index]

        #VALUTAZIONE
        loss, mae = model.evaluate(X_test, y_test, verbose=1)
        metrics.append({'loss': loss, 'mae': mae})

    #MEDIA E DEVIAZIONE STANDARD
    avg_loss = np.mean([m['loss'] for m in metrics])
    avg_mae = np.mean([m['mae'] for m in metrics])
    std_loss = np.std([m['loss'] for m in metrics])
    std_mae = np.std([m['mae'] for m in metrics])

    print(f"\nRisultati per {target_name}:")
    print(f"Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    return target_name, avg_loss, std_loss, avg_mae, std_mae


#PER OGNI TARGET ESEGUIAMO EVALUTATE_MODEL_WITH_TSCV USANDO LE FEATURE ASSOCIATE, SALVANDO I RISULTATI
all_results = []
for target_name, feature_columns in features_dict.items():
    model_dir = os.path.join(models_dir, target_name)
    results = evaluate_model_with_tscv(
        target_name=target_name,
        features=feature_columns,
        model_path=os.path.join(model_dir, f"{target_name}_model.keras"),
        scaler_X_path=os.path.join(model_dir, "scaler_X.pkl"),
        scaler_y_path=os.path.join(model_dir, "scaler_y.pkl")
    )
    all_results.append(results)

#STAMPA FINALE DI TUTTI I RISULTATI
print("\n\nRiepilogo finale dei risultati:")
for result in all_results:
    target_name, avg_loss, std_loss, avg_mae, std_mae = result
    print(f"Target: {target_name}")
    print(f"Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}\n")
