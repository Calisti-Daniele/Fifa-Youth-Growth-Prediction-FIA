import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carica il dataset
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23.csv')

# 1. Analisi iniziale
print("Dimensione del dataset:", df.shape)
print("Prime righe del dataset:\n", df.head())
print("Informazioni sul dataset:\n", df.info())
print("Statistiche descrittive:\n", df.describe())

# 2. Gestione dei valori nulli
# Controlla i valori nulli
print("Valori nulli per colonna:\n", df.isnull().sum())

# Sostituisci valori nulli con la media (opzione 2)
# Se vuoi sostituire i valori nulli, ad esempio, per colonne numeriche:
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].mean(), inplace=True)

# 3. Normalizzazione dei dati
# Seleziona le colonne numeriche per la normalizzazione
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Crea un oggetto StandardScaler
scaler = StandardScaler()

# Normalizza i dati
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 4. Analisi finale
print("Dopo il preprocessing:")
print("Valori nulli per colonna:\n", df.isnull().sum())
print("Prime righe del dataset preprocessato:\n", df.head())

# Salva il dataset preprocessato in un nuovo file CSV
df.to_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed.csv', index=False)
print("Dataset preprocessato salvato come 'dataset_fifa_15_23_preprocessed.csv'")

print(df.describe())