"""
Questo script filtra i file CSV contenenti dati sui giocatori di calcio per selezionare
solo quelli con età ≤ 23 anni e valore complessivo (`overall`) ≤ 70.
I file filtrati vengono salvati in una cartella separata per una successiva analisi o utilizzo.

Funzionalità principali:
- Lettura di file CSV corrispondenti al pattern specificato.
- Applicazione di filtri per età e valore complessivo.
- Salvataggio dei file filtrati con un suffisso `_filtered`.
- Monitoraggio del numero di righe originali e filtrate per ogni file.
"""


import pandas as pd
import glob
import os

#Percorso della cartella contenente i file CSV originali
input_path = '../datasets/players_*.csv'

#Percorso della cartella in cui salvare i file filtrati
output_dir = '../datasets/filtered/'

#CREA LA CARTELLA DI DESTINAZIONE SE NON ESISTE
os.makedirs(output_dir, exist_ok=True)

#Lista dei file CSV che corrispondono al pattern
files = glob.glob(input_path)

# Ciclo attraverso ogni file CSV
for file in files:

    #LEGGI IL FILE CSV
    df = pd.read_csv(file)

    #Filtriamo il DataFrame per mantenere solo le righe dove 'age' <= 23
    df_filtered = df[(df['age'] <= 23) & (df['overall'] <= 70)]

    #CostruiAMO il nome del nuovo file CSV per il salvataggio
    filtered_file_name = os.path.join(output_dir, os.path.basename(file).replace('.csv', '_filtered.csv'))

    #Salviamo il DataFrame filtrato in un nuovo file CSV
    df_filtered.to_csv(filtered_file_name, index=False)

    #Stampiamo il numero di righe nel file originale e nel file filtrato
    original_rows = df.shape[0]
    filtered_rows = df_filtered.shape[0]
    print(f"File: {file} - Righe originali: {original_rows}, Righe filtrate: {filtered_rows}")
