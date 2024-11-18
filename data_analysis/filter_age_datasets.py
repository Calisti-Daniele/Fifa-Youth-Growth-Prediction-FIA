import pandas as pd
import glob
import os

# Percorso della cartella contenente i file CSV originali
input_path = '../datasets/players_*.csv'
# Percorso della cartella in cui salvare i file filtrati
output_dir = '../datasets/filtered/'

# Crea la cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

# Lista dei file CSV che corrispondono al pattern
files = glob.glob(input_path)

# Ciclo attraverso ogni file CSV
for file in files:
    # Leggi il file CSV
    df = pd.read_csv(file)

    # Filtra il DataFrame per mantenere solo le righe dove 'age' <= 23
    df_filtered = df[(df['age'] <= 23) & (df['overall'] <= 70)]

    # Costruisci il nome del nuovo file CSV per il salvataggio
    filtered_file_name = os.path.join(output_dir, os.path.basename(file).replace('.csv', '_filtered.csv'))

    # Salva il DataFrame filtrato in un nuovo file CSV
    df_filtered.to_csv(filtered_file_name, index=False)

    # Stampa il numero di righe nel file originale e nel file filtrato
    original_rows = df.shape[0]
    filtered_rows = df_filtered.shape[0]
    print(f"File: {file} - Righe originali: {original_rows}, Righe filtrate: {filtered_rows}")
