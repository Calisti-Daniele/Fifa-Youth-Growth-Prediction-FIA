import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carica il dataset
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23.csv')
df_not_normalized = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23.csv')

# 1. Analisi iniziale
print("Dimensione del dataset:", df.shape)
print("Prime righe del dataset:\n", df.head())
print("Informazioni sul dataset:\n", df.info())
print("Statistiche descrittive:\n", df.describe())

# 2. Gestione dei valori nulli
# Controllo dei valori nulli
print("Valori nulli per colonna:\n", df.isnull().sum())

# Sostituzione dei valori nulli con la media per colonne numeriche
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].mean(), inplace=True)
    df_not_normalized[column].fillna(df_not_normalized[column].mean(), inplace=True)

# 3. Normalizzazione dei dati
# Seleziono le colonne numeriche per la normalizzazione
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Creo un oggetto StandardScaler
scaler = StandardScaler()

# Normalizzo i dati
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 4. Encoding binario per 'preferred_foot'
# Converto 'preferred_foot' in valori binari: 0 = Left, 1 = Right
df['preferred_foot'] = df['preferred_foot'].map({'Left': 0, 'Right': 1})

# 5. Analisi finale
print("Dopo il preprocessing:")
print("Valori nulli per colonna:\n", df.isnull().sum())
print("Prime righe del dataset preprocessato:\n", df.head())

# Salvo il dataset preprocessato in un nuovo file CSV
df.to_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed.csv', index=False)
df_not_normalized.to_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed_not_normalized.csv', index=False)
print("Dataset preprocessato salvato come 'dataset_fifa_15_23_preprocessed_not_normalized.csv'")
