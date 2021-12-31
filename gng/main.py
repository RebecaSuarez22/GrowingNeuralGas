import tensorflow as tf
import pandas as pd
from GrowingNeuralGas import GrowingNeuralGas
from sklearn.preprocessing import StandardScaler


def test():
    tf.random.set_seed(23)
    X = tf.concat([tf.random.normal([500, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([0.0, 0.0]),
                   tf.random.normal([500, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([1.0, 0.0]),
                   tf.random.normal([500, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([1.0, 1.0])], 0)

    data = pd.read_csv("Sample_Cluster_Data_2D.csv")
    t = StandardScaler().fit_transform(data.loc[:100, list(data.columns)].values)
    X = tf.constant(t, dtype=tf.float32)
    growingNeuralGas = GrowingNeuralGas()
    growingNeuralGas.fit(X, 5)
    pass


def clasificación_vinos():
    data = pd.read_csv("Clasificación_Vinos.csv")
    t = StandardScaler().fit_transform(data.loc[:100, list(data.columns)].values)
    X = tf.constant(t, dtype=tf.float32)
    gng = GrowingNeuralGas()
    gng.fit(X, 5)





if __name__ == '__main__':
    clasificación_vinos()
