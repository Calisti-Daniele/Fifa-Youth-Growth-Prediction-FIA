import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset
file_path = 'outputs/fia_comparison_2024.csv'  # Sostituisci con il percorso del tuo file
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

# Creazione del grafico
plt.figure(figsize=(12, 6))
percentage_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribuzione percentuale delle differenze')
plt.xlabel('Intervallo di Differenza')
plt.ylabel('Percentuale (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Generazione del testo scritto
text_output = "\nDistribuzione percentuale delle differenze:\n"
for range_label, percentage in percentage_distribution.items():
    text_output += f"{range_label} -> {percentage:.2f}%\n"

# Stampa del testo
print(text_output)
