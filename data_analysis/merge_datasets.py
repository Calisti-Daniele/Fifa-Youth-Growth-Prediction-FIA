"""
Questo script unisce più file CSV filtrati in un unico dataset consolidato, mantenendo solo le colonne di interesse.

Funzionamento:
1. Cerca tutti i file CSV nella directory specificata (`../datasets/filtered/`) che corrispondono al pattern `players_*_filtered.csv`.
2. Per ogni file trovato:
   - Carica il file CSV in un DataFrame.
   - Filtra il DataFrame per mantenere solo un set predefinito di colonne utili.
   - Aggiunge il DataFrame filtrato a una lista per l'unione successiva.
3. Combina tutti i DataFrame in un unico dataset grande.
4. Salva il dataset risultante in un nuovo file CSV (`../datasets/ready_to_use/dataset_fifa_15_23.csv`).
5. Stampa il numero di righe per ogni file e il totale complessivo.

Utilità:
- Questo script è utile per consolidare dati di giocatori FIFA filtrati, riducendo il dataset solo alle informazioni necessarie
  e preparando i dati per ulteriori analisi o modelli di machine learning.
"""


import pandas as pd
import glob

#Percorso della cartella contenente i file CSV
path = '../datasets/filtered/players_*_filtered.csv'

#Lista dei file CSV che corrispondono al pattern
files = glob.glob(path)

#COLONNE DA SELEZIONARE
columns_to_select = [
    'player_url',
    'short_name',
    'long_name',
    'player_positions',
    'overall',
    'potential',
    'age',
    'height_cm',
    'weight_kg',
    'club_name',
    'league_name',
    'nationality_name',
    'preferred_foot',
    'shooting',
    'passing',
    'dribbling',
    'defending',
    'physic',
    'attacking_crossing',
    'attacking_finishing',
    'attacking_heading_accuracy',
    'attacking_short_passing',
    'attacking_volleys',
    'skill_dribbling',
    'skill_curve',
    'skill_fk_accuracy',
    'skill_long_passing',
    'skill_ball_control',
    'movement_acceleration',
    'movement_sprint_speed',
    'movement_agility',
    'movement_reactions',
    'movement_balance',
    'power_shot_power',
    'power_jumping',
    'power_stamina',
    'power_strength',
    'power_long_shots',
    'mentality_aggression',
    'mentality_interceptions',
    'mentality_positioning',
    'mentality_vision',
    'mentality_penalties',
    'mentality_composure',
    'defending_marking_awareness',
    'defending_standing_tackle',
    'defending_sliding_tackle',
    'goalkeeping_diving',
    'goalkeeping_handling',
    'goalkeeping_kicking',
    'goalkeeping_positioning',
    'goalkeeping_reflexes',
    'goalkeeping_speed',
    'fifa_version'
]

#Lista per conservare i DataFrame
dataframes = []
total_rows = 0  #Variabile per contare il numero totale di righe

#Usiamo glob.glob(path) per ottenere una lista di file CSV corrispondenti al pattern
#e itera attraverso ciascun file.
for file in files:
    #LEGGE IL FILE CSV
    df = pd.read_csv(file)

    #Seleziona solo le colonne di interesse
    df_selected = df[columns_to_select]

    #Stampa il numero di righe per il dataset corrente
    num_rows = df_selected.shape[0]
    print(f"Numero di righe nel file {file}: {num_rows}")

    #Aggiungiamo il DataFrame selezionato alla lista
    dataframes.append(df_selected)

    #Aggiornaiamo il numero totale di righe
    total_rows += num_rows

#COMBINA TUTTI I DATAFRAME ACCUMULATI IN UN UNICO DATAFRAME GRANDE
final_df = pd.concat(dataframes, ignore_index=True)

#SALVIAMO IL DATAGRAME FINALE IN UN NUOVO FILE CSV
final_df.to_csv('../datasets/ready_to_use/dataset_fifa_15_23.csv', index=False)

#STAMPA IL NUMERO TOTALE DI RIGHE
print(f"Numero totale di righe: {total_rows}")
