#Questo script legge e analizza vari file CSV contenenti differenze
#tra predizioni e valori reali per diversi aspetti del gioco (ad esempio, "shooting")
#Per ciascun file:
#Viene calcolata la distribuzione delle differenze.
#Viene creato un grafico a barre che mostra la distribuzione percentuale delle differenze.
#Vengono generati e stampati messaggi che riepilogano le percentuali di errore, inclusi i
#giocatori che hanno un errore inferiore a 1 punto.
#L'obiettivo è fornire una visione chiara e visuale delle differenze tra le predizioni e
#i valori reali per diversi aspetti di un gioco (probabilmente nel contesto di dati relativi
#a giocatori di calcio o altri sport).

import pandas as pd #GESTIONE DEI DATI IN FORMATO TABELLARE
import matplotlib.pyplot as plt #CREAZIONE DEI GRAFICI E VISUALIZZAZIONE DEI DATI
import os #USATO PER INTERAGIRE CON IL SISTEMA OPERATIVO

#PERCORSI DEI FILE CSV CHE CONTENGONO I DATI DA ANALIZZARE
files = [
    'outputs/fia_comparison_overall.csv',
    'outputs/fia_comparison_shooting.csv',
    'outputs/fia_comparison_passing.csv',
    'outputs/fia_comparison_dribbling.csv',
    'outputs/fia_comparison_defending.csv',
    'outputs/fia_comparison_Physicality.csv'
]

#LISTA CHE VIENE CREATA PER CONSERVARE I MESSAGGI CHE VERRANNO GENERATI ALLA FINE DELL'ANALISI
final_messages = []

#FUNZIONE PRINCIPLAE CHE ANALIZZA I DATI CONTENUTI IN UN FILE CSV SPECIFICO.
def analyze_differences(file_path, target):
    #CARICA IL FILE CSV E I DATI VENGONO LETTI
    data = pd.read_csv(file_path)

    #CALCOLO DEL VALORE MAX DELLA COLONNA 'DIFFERENZA'
    max_difference = int(data['Differenza'].max())
    bins = list(range(0, max_difference + 2))  # Intervalli da 0 a max_difference
    labels = [f"{i}-{i+1}" for i in bins[:-1]]

    #OGNI VALORE DELLA COLONNA 'DIFFERENZA' VIENE ASSEGNATA A UNO DEGLI INTERVALLI CREATI
    data['Difference Range'] = pd.cut(data['Differenza'], bins=bins, labels=labels, right=False)

    #CALCOLO DELLA PERCENTUALE PER OGNI INTERVALLO IN PERCENTUALE
    percentage_distribution = data['Difference Range'].value_counts(normalize=True) * 100

    #ORDINIAMO I DATI PER RANGE
    percentage_distribution = percentage_distribution.sort_index()

    #Calcolo della percentuale totale per errori tra 0 e 5
    error_below_5 = percentage_distribution.loc[percentage_distribution.index[:1]].sum()
    total_error_message = f"Percentuale di giocatori la cui predizione per {target} sbaglia di max. 1 punti: {error_below_5:.2f}%"
    final_messages.append(total_error_message)

    #CREAZIONE DEL GRAFICO
    plt.figure(figsize=(12, 6))
    percentage_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Distribuzione percentuale delle differenze - {target}')
    plt.xlabel('Intervallo di Differenza')
    plt.ylabel('Percentuale (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    #GENERAZIONE DEL TESTO SCRITTO
    text_output = f"\nDistribuzione percentuale delle differenze per {target}:\n"
    for range_label, percentage in percentage_distribution.items():
        text_output += f"{range_label} -> {percentage:.2f}%\n"
    text_output += f"\n{total_error_message}\n"

    #STAMPA DEL TESTO
    print(text_output)

#LISTA CHE CONTIENE I NOMI DEGLI ASPETTI CHE VENGONO ANALIZZATI
targets = ['overall', 'shooting', 'passing', 'dribbling', 'defending', 'Physicality']

#PER OGNI FILE VIENE VERIFICATO SE IL FILE ESISTE
for file, target in zip(files, targets):
    if os.path.exists(file):
        analyze_differences(file, target)
    else:
        print(f"File non trovato: {file}")

#STAMPA DEI MESSAGGI FINALI
print("\n--- Conclusioni ---")
for message in final_messages:
    print(message)
