import pandas as pd

df = pd.read_csv('datasets/dataset_fc_24.csv')


# Aggiungi le colonne 'experience' e 'age_trend'
df = df.sort_values('long_name')
df['experience'] = 1
df['age_trend'] = df['Age'].diff().fillna(0)

df.to_csv("datasets/dataset_fc_24.csv", index=False)
