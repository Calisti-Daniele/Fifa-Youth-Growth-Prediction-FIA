#tensorflow è una libreria open-source per il ML, particolarmente utilizzata per
#costruire e allenare modelli di deep learning
import tensorflow as tf

#keras è un'interfaccia di alto livello per Tensorflow che semplifica la costruzione
#di modelli di deep learning
import keras

import pandas as pd #libreria per la manipolazione dei dati, in particolare sui DataFrame


#weighted_loss è una funzione di perdita personalizzata che calcola l'errore tra i valori
#reali y_true e i valori previsti y_pred, ma con un peso che dipende dalla distanza tra i due
@keras.utils.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    weights = 1 / (1 + tf.abs(y_true - y_pred))  # Più lontani sono i valori, maggiore sarà il peso assegnato
    #la funzione di perdita finale è la media del prodotto del peso e del quadrato della
    #differenza tra y_true e y_pred. Si usa il quadrato della differenza per penalizzare
    #gli errori più grandi in modo più severo
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

#funzione per caricare ed elaborare il dataset
def load_dataset(dataset_path):
    #carica il dataset da un file CSV utilizzando la funzione read_csv di pandas,
    #creando un dataframe df contenente i dati
    df = pd.read_csv(dataset_path)

    # AGGIUNGIAMO:
    # EXPERIENCE: INDICA L'ESPERIENZA DEL GIOCATOR, CONTA QUANTE VOLTE UN GIOCATORE APPARE NEL DATASET
    df['experience'] = df.groupby('long_name').cumcount() + 1
    # AGE_TREND: CALCOLA LA DIFFERENZA DI ETÀ TRA I VARI RECORD DELLO STESSO GIOCATORE, PER OSSERVARE
    # COME CAMBIA L'ETÀ TRA I DIVERSI ANNI
    df['age_trend'] = df['age'].diff().fillna(0)

    #restituiamo il dataframe modificato con le nuove colonne experience e age_trend
    return df