# Questo script esegue operazioni di preprocessing su un dataset contenente dati sui giocatori di FIFA
# Le operazioni principali includono:
# 1. Caricamento del dataset da un file CSV.
# 2. Analisi preliminare dei dati (dimensioni, prime righe, informazioni generali e statistiche descrittive).
# 3. Gestione dei valori nulli:
#    - I valori nulli nelle colonne numeriche vengono sostituiti con la media della colonna.
# 4. Normalizzazione dei dati numerici:
#    - Le colonne numeriche vengono normalizzate utilizzando StandardScaler, che ridimensiona i dati per avere media 0 e deviazione standard 1.
# 5. Encoding della colonna 'preferred_foot':
#    - La colonna 'preferred_foot' (piede preferito) viene trasformata in valori binari (0 = sinistro, 1 = destro).
# 6. Analisi finale dei dati dopo il preprocessing:
#    - Viene verificato che non ci siano più valori nulli e viene visualizzato un campione delle prime righe.
# 7. Salvataggio del dataset preprocessato in due file CSV:
#    - 'dataset_fifa_15_23_preprocessed.csv': il dataset con i dati normalizzati.
#    - 'dataset_fifa_15_23_preprocessed_not_normalized.csv': il dataset senza normalizzazione.

import pandas as pd
from sklearn.preprocessing import StandardScaler

#CARICHIAMO IL DATASET
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23.csv')
df_not_normalized = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23.csv')

#ANALISI PRELIMINARE DEL DATASET
print("Dimensione del dataset:", df.shape) #DIMENSIONE DEL DATASET
print("Prime righe del dataset:\n", df.head()) #MOSTRA LE PRIME RIGHE DEL DATASET
print("Informazioni sul dataset:\n", df.info()) #FORNISCE INFORMAZIONI SUI TIPI DI DATO DI CIASCUNA COLONNA
print("Statistiche descrittive:\n", df.describe()) #FORNISCE STATISTICHE DESCRITTIVE PER TUTTE LE COLONNE NUMERICHE

#GESTIONE DEI VALORI NULLI
#Controllo dei valori nulli
print("Valori nulli per colonna:\n", df.isnull().sum())

#Sostituzione dei valori nulli con la media per colonne numeriche
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].mean(), inplace=True)
    df_not_normalized[column].fillna(df_not_normalized[column].mean(), inplace=True)

#NORMALIZZAZIONE DEI DATI
#Selezioniamo le colonne numeriche per la normalizzazione
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

#Creiamo un oggetto StandardScaler
scaler = StandardScaler()

#NORMALIZZIAMO I DATI
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#Encoding binario per 'preferred_foot'
#Convertiamo 'preferred_foot' in valori binari: 0 = Left, 1 = Right
df['preferred_foot'] = df['preferred_foot'].map({'Left': 0, 'Right': 1})

#Analisi finale
print("Dopo il preprocessing:")
print("Valori nulli per colonna:\n", df.isnull().sum())
print("Prime righe del dataset preprocessato:\n", df.head())

#Salviamo il dataset preprocessato in un nuovo file CSV
df.to_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed.csv', index=False)
df_not_normalized.to_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv', index=False)
print("Dataset preprocessato salvato come 'dataset_fifa_15_23_preprocessed_not_normalized.csv'")
