import tensorflow as tf
import keras
import pandas as pd


@keras.utils.register_keras_serializable()
def weighted_loss(y_true, y_pred):
    weights = 1 / (1 + tf.abs(y_true - y_pred))  # Più lontano, più peso
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)

    # Aggiunta di feature extra
    df['experience'] = df.groupby('long_name').cumcount() + 1
    df['age_trend'] = df['age'].diff().fillna(0)

    return df
