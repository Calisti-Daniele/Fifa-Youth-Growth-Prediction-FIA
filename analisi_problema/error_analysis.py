#Il codice carica un dataset, calcola la distribuzione percentuale delle
#differenze (in base agli intervalli), visualizza un grafico delle percentuali
#e stampa la distribuzione sotto forma di testo.


import pandas as pd #libreria utilizzata per manipolare e analizzare i dati in formato tabellare
import matplotlib.pyplot as plt #libreria per la creazione di grafici e visualizzazioni in PY

#CARICAMENTO DEL DATASET
file_path = 'outputs/fia_comparison_2024.csv' #PERCORSO DEL FILE CSV CONTENENTE IL DATASET
data = pd.read_csv(file_path) #DENTRO IL DATAFRAME DATA CARICHIAMO IL DATASET



#Calcolo delle percentuali delle differenze per intervalli di 1 unità
max_difference = int(data['Differenza'].max()) #CALCOLA IL VALORE MAX NELLA COLONNA DIFFERENZA
bins = list(range(0, max_difference + 2))  #CREA UNA LISTA DI INTERVALLI DA 0 FINO AL MAX VALORE DELLA DIFFERENZA +1
labels = [f"{i}-{i+1}" for i in bins[:-1]] #CREA UNA LISTA DI ETICHETTE PER CIASCUN INTERVALLO


#SUDDIVIDIAMO I VALORI DELLA COLONNA DIFFERENZA IN INTERVALLI DEFINITI IN BINS, ASSEGNANDO
#AD OGNI VALORE L'INTERVALLO CORRISPONDENTE
data['Difference Range'] = pd.cut(data['Differenza'], bins=bins, labels=labels, right=False)

#CALCOLIAMO LA PERCENTUALE DI OGNI INTERVALLO ALL'INTERNO DELLA COLONNA DIFFERENCE RANGE
percentage_distribution = data['Difference Range'].value_counts(normalize=True) * 100



#ORDINIAMO I DATI PER RANGE, DAL PIÙ BASSO AL PIÙ ALTO
percentage_distribution = percentage_distribution.sort_index()

#CREAZIONE DEL GRAFICO
plt.figure(figsize=(12, 6)) #DIMENSIONE DEL GRAFICO
percentage_distribution.plot(kind='bar', color='skyblue', edgecolor='black') #GRAFICO A BARRE
plt.title('Distribuzione percentuale delle differenze') #TITOLO
plt.xlabel('Intervallo di Differenza') #ETICHETTA ASSE X
plt.ylabel('Percentuale (%)') #ETICHETTA ASSE Y
plt.xticks(rotation=45) #RUOTA LE ETICHETTE DI 45 GRADI PER FACILITARE LA LETTURA DEL GRAFICO
plt.tight_layout() #OTIMIZZA LA DISPOSIZIONE DEGLI ELEMENTI NEL GRAFICO
plt.show() #VISUALIZZA IL GRAFICO

#Generazione del testo scritto
text_output = "\nDistribuzione percentuale delle differenze:\n"
for range_label, percentage in percentage_distribution.items():
    #PER OGNI INTERVALLO E LA CORRISPONDENTE PERCENTUALE NELLA DISTRIBUZIONE,
    #VIENE AGGIUNTO UN RIGO AL TESTO NEL SEGUENTE FORMATO
    text_output += f"{range_label} -> {percentage:.2f}%\n"

#STAMPA DEL TESTO GENERATO
print(text_output)
