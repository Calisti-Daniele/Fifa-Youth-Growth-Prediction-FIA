"""
Questo script esegue un'analisi delle feature per prevedere più target contemporaneamente
(overall, potential, shooting, passing, dribbling, defending, physic) utilizzando un approccio multivariato.

Passaggi principali:
1. Separazione delle feature indipendenti (X) e dei target da prevedere (y):
   - X: Tutte le colonne utili eccetto i target e altre informazioni non rilevanti.
   - y: Colonne target da prevedere.
2. Calcolo della matrice di correlazione tra feature e target:
   - Analisi della correlazione di Pearson per identificare l'impatto delle feature su ciascun target.
3. Visualizzazione dei risultati:
   - Heatmap della correlazione per mostrare graficamente le relazioni tra feature e target.
   - Supporto alla selezione delle feature più rilevanti per migliorare le prestazioni del modello.
"""

'''
    Per effettuare la feature selection con l'obiettivo di prevedere più target contemporaneamente
    (overall, potential, shooting, passing, dribbling, defending, physic)
    Possiamo scegliere di seguire un approccio specifico per l'analisi multivariata.
'''
from training_models.functions import load_dataset #carichiamo il dataset pre-elaborato

'''
    1. Separare le feature e i target
    Dividiamo il dataset in:
    Feature indipendenti (X): Tutte le colonne tranne i target da prevedere.
    Feature dipendenti (y): I target che vogliamo prevedere.
'''
import seaborn as sns
import matplotlib.pyplot as plt

#CARICAMENTO DEL DATASET
df = load_dataset('../datasets/ready_to_use/dataset_fifa_15_23_preprocessed.csv')

#DEFINIAMO I TARGET
target_columns = ['overall', 'potential', 'shooting', 'passing', 'dribbling', 'defending', 'physic']

#RIMUOVIAMO I TARGET DA PREVEDERE E LE INFORMAZIONI NON UTILI PER LE PREVISIONI
X = df.drop(columns=target_columns + ['player_url','player_positions', 'short_name', 'long_name', 'club_name', 'league_name', 'nationality_name', 'fifa_version'])
y = df[target_columns]

'''
    2. Analisi delle correlazioni multivariata
    Per capire l'impatto di ogni feature sui target:
        Matrice di correlazione estesa (correlazione di Pearson):
        Mostrare le correlazioni tra le feature e ciascun target.
'''

#CALCOLIAMO LA MATRICE DI CORRELAZIONE
corr_matrix = df[target_columns + list(X.columns)].corr()

#VISUALIZZAZIONE DELLA CORRELAZIONE
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix[target_columns], annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlazione tra feature e target")
plt.show()

