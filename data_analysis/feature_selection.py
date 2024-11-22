'''
    Per effettuare la feature selection con l'obiettivo di prevedere più target contemporaneamente
    (overall, potential, shooting, passing, dribbling, defending, physic)
    Possiamo scegliere di seguire un approccio specifico per l'analisi multivariata.
'''

'''
    1. Separare le feature e i target
    Dividiamo il dataset in:
    Feature indipendenti (X): Tutte le colonne tranne i target da prevedere.
    Feature dipendenti (y): I target che vogliamo prevedere.
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carico il dataset
df = pd.read_csv('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed.csv')

# Definisco i target
target_columns = ['overall', 'potential', 'shooting', 'passing', 'dribbling', 'defending', 'physic']

# Divido feature e target

#Rimuovo i target da prevedere e le info non utili ai fini della previsione
X = df.drop(columns=target_columns + ['player_url','player_positions', 'short_name', 'long_name', 'club_name', 'league_name', 'nationality_name', 'fifa_version'])
y = df[target_columns]

'''
    2. Analisi delle correlazioni multivariata
    Per capire l'impatto di ogni feature sui target:
        Matrice di correlazione estesa (correlazione di Pearson):
        Mostrare le correlazioni tra le feature e ciascun target.
'''

# Calcola la matrice di correlazione
corr_matrix = df[target_columns + list(X.columns)].corr()

# Visualizza la correlazione tra le feature e i target
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix[target_columns], annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlazione tra feature e target")
plt.show()

