#Il codice esegue una serie di operazioni per effettuare predizioni su
#diversi target (come il punteggio complessivo, il tiro, il passaggio, ecc.)
#per i giocatori FIFA. Utilizza un modello pre-allenato per fare queste predizioni
#basate su sequenze temporali di dati (5 anni consecutivi per ciascun giocatore),
#confronta le predizioni con i valori reali del dataset FC 24 (2024), e
#salva i risultati in file CSV per ciascun target.

import os
import pickle #CARICA E SALVA OGGETTI PY SERIALIZZATI
import numpy as np #PER OPERAZIONI NUMERICHE AVANZATE
from training_models.functions import * #IMPORTA TUTTE LE FUNZIONI DEFINITE NEL MODULO FUNCTIONS

#PERCORSO DEI FILE

#PERCORSO AL FILE CSV CON I DATI PRE-ELABORATI DAL 2015 AL 2023, NON NORMALIZZATI
dataset_path = '../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv'
#PERCORSO AL FILE CSV CONTENENTE I DATI DEI GIOCATORI PER IL 2024
dataset_fc_24_path = '../datasets/dataset_fc_24.csv'
#PERCORSO ALLA CARTELLA CHE CONTIENE I MODELLI E GLI SCALER PRE-ALLENATI
models_dir = '../models/'

#CARICAMENTO DEI DATASET
df = load_dataset(dataset_path)
df = df.sort_values(by=['long_name', 'fifa_version'])

#CARICAMENTO DEL DATASET DI FC24
df_fc_24 = pd.read_csv(dataset_fc_24_path)

#AGGIUNTA DI DUE COLONNE

#EXPERIENCE: INDICA L'ESPERIENZA DEL GIOCATOR, CONTA QUANTE VOLTE UN GIOCATORE APPARE NEL DATASET
df_fc_24['experience'] = df_fc_24.groupby('long_name').cumcount() + 1
#AGE_TREND: CALCOLA LA DIFFERENZA DI ETÀ TRA I VARI RECORD DELLO STESSO GIOCATORE, PER OSSERVARE
#COME CAMBIA L'ETÀ TRA I DIVERSI ANNI
df_fc_24['age_trend'] = df_fc_24.groupby('long_name')['Age'].diff().fillna(0)

#QUI VENGONO DEFINITI I TARGET E LE FEATURE ASSOCIATE.
#OGNI TARGET HA UNA LISTA DI FEATURE, OVVERO LE COLONNE DEL DATASET CHE VERRANNO
#UTILIZZATE PER EFFETTUARE LA PREVISIONE.
features = {
    'overall': ['potential', 'passing', 'dribbling', 'movement_reactions', 'mentality_composure'],
    'shooting': ['passing', 'dribbling', 'attacking_finishing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
                 'skill_long_passing', 'skill_ball_control', 'movement_agility', 'power_shot_power', 'power_long_shots',
                 'mentality_positioning', 'mentality_vision', 'mentality_penalties'],
    'passing': ['shooting', 'dribbling', 'attacking_crossing', 'attacking_short_passing', 'skill_dribbling',
                'skill_curve', 'skill_long_passing', 'skill_fk_accuracy', 'skill_ball_control', 'power_long_shots',
                'mentality_vision', 'mentality_positioning'],
    'dribbling': ['shooting', 'passing', 'attacking_crossing', 'attacking_finishing', 'attacking_volleys',
                  'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_ball_control', 'movement_acceleration',
                  'movement_agility', 'movement_balance', 'power_long_shots', 'mentality_positioning',
                  'mentality_vision'],
    'defending': ['physic', 'mentality_aggression', 'mentality_interceptions', 'defending_marking_awareness',
                  'defending_standing_tackle', 'defending_sliding_tackle'],
    'physic': ['defending', 'power_strength', 'mentality_aggression', 'mentality_interceptions']
}

#INDICA LA LUNGHEZZA DELLA SEQUENZA TEMPORALE UTILIZZATA PER FARE LA PREVISIONE
#IL MODELLO GUARDERÀ GLI ULTIMI 5 VALORI TEMPORALI DI CIASCUN GIOCATORE PER FARE LA PREVISIONE
timesteps = 5

#FUNZIONE PER EFFETTUARE PREDIZIONI E PER SALVARE I RISULTATI
def predict_and_save_results(target, feature_columns):
    print(f"Inizio predizioni per il target: {target}")

    #CARICAMENTO DEL MODELLO E DEGLI SCALER
    model_path = os.path.join(models_dir, target, f"{target}_model.keras")
    scaler_X_path = os.path.join(models_dir, target, "scaler_X.pkl")
    scaler_y_path = os.path.join(models_dir, target, "scaler_y.pkl")

    model = keras.models.load_model(model_path, custom_objects={"weighted_loss": weighted_loss})
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    #PREPARIAMO I DATI DI INPUT PER LA PREVISIONE
    X, player_names = [], []
    for player, player_data in df.groupby('long_name'):
        if len(player_data) >= timesteps + 1:
            player_data = player_data[feature_columns].values
            X.append(player_data[-timesteps:, :])  # Prendi le ultime `timesteps`
            player_names.append(player)  # Salva il nome del giocatore

    X = np.array(X)  # Shape: (num_players, timesteps, num_features)
    print(X.shape)

    #NORMALIZZIAMO I DATI
    X_flat = X.reshape(-1, len(feature_columns))
    X_scaled = scaler_X.transform(X_flat).reshape(X.shape)
    y_pred_scaled = model.predict(X_scaled)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

    #CONFRONTIAMO LE PREDIZIONI CON I VALORI EFFETTI DEL 2024
    results = []
    for player, pred_value in zip(player_names, y_pred_original.flatten()):

        if target == "physic":
            target = "Physicality"

        actual_value = df_fc_24[df_fc_24['long_name'] == player][target.capitalize()].values
        if actual_value.size > 0:
            actual_value = actual_value[0]
        else:
            actual_value = None

        results.append({
            'Nome': player,
            'Target': target,
            'Previsto': pred_value,
            'Effettivo (2024)': actual_value,
            'Differenza': abs(pred_value - actual_value) if actual_value is not None else None
        })

    #CREIAMO UN DATAFRAME PER IL CONFRONTO
    comparison_df = pd.DataFrame(results)

    #ORDINIAMO PER DIFFERENZA DECRESCENTE
    comparison_df = comparison_df.sort_values(by='Differenza', ascending=False)

    #SALVIAMO IL DATAFRAME
    output_path = f'outputs/fia_comparison_{target}.csv'
    comparison_df.to_csv(output_path, index=False)
    print(f"Confronto salvato in '{output_path}'\n")

#EFFETTUIAMO LE PREDIZIONI PER OGNI TARGET
for target_name, feature_cols in features.items():
    predict_and_save_results(target_name, feature_cols)
