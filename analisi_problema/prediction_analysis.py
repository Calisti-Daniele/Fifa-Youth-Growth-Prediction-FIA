import pandas as pd
import matplotlib.pyplot as plt
import os

# Percorsi dei file dei confronti
files = [
    'outputs/fia_comparison_overall.csv',
    'outputs/fia_comparison_shooting.csv',
    'outputs/fia_comparison_passing.csv',
    'outputs/fia_comparison_dribbling.csv',
    'outputs/fia_comparison_defending.csv',
    'outputs/fia_comparison_Physicality.csv'
]

# Lista per conservare i messaggi finali
final_messages = []

# Funzione per calcolare e visualizzare le percentuali delle differenze per un file specifico
def analyze_differences(file_path, target):
    # Carica il dataset
    data = pd.read_csv(file_path)

    # Calcolo delle percentuali delle differenze per intervalli di 1 unità
    max_difference = int(data['Differenza'].max())
    bins = list(range(0, max_difference + 2))  # Intervalli da 0 a max_difference
    labels = [f"{i}-{i+1}" for i in bins[:-1]]

    data['Difference Range'] = pd.cut(data['Differenza'], bins=bins, labels=labels, right=False)

    # Calcolo della percentuale per ciascun intervallo
    percentage_distribution = data['Difference Range'].value_counts(normalize=True) * 100

    # Ordina i dati per range
    percentage_distribution = percentage_distribution.sort_index()

    # Calcolo della percentuale totale per errori tra 0 e 5
    error_below_5 = percentage_distribution.loc[percentage_distribution.index[:1]].sum()
    total_error_message = f"Percentuale di giocatori la cui predizione per {target} sbaglia di max. 1 punti: {error_below_5:.2f}%"
    final_messages.append(total_error_message)

    # Creazione del grafico
    plt.figure(figsize=(12, 6))
    percentage_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Distribuzione percentuale delle differenze - {target}')
    plt.xlabel('Intervallo di Differenza')
    plt.ylabel('Percentuale (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Generazione del testo scritto
    text_output = f"\nDistribuzione percentuale delle differenze per {target}:\n"
    for range_label, percentage in percentage_distribution.items():
        text_output += f"{range_label} -> {percentage:.2f}%\n"
    text_output += f"\n{total_error_message}\n"

    # Stampa del testo
    print(text_output)

# Elenco dei target per i file
targets = ['overall', 'shooting', 'passing', 'dribbling', 'defending', 'Physicality']

# Analizza tutti i file
for file, target in zip(files, targets):
    if os.path.exists(file):
        analyze_differences(file, target)
    else:
        print(f"File non trovato: {file}")

# Stampa i messaggi finali
print("\n--- Conclusioni ---")
for message in final_messages:
    print(message)
