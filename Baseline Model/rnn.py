import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import data_processing

class RNN_model:
    def __init__(self, data):
        inputs = keras.Input(shape = [None, frames, input_size])
        model = tf.keras.layers.LSTM(64, dropout = 0.2, return_sequences = True)(inputs)
        model = tf.keras.layers.LSTM(64, dropout = 0.2, return_sequences = True)(model)
        model = tf.keras.layers.LSTM(64, dropout = 0.2, return_sequences = True)(model)
        model = tf.keras.Dense(data.get_num_classes(), activation = 'softmax')
        model.compile(optimizer = 'Adam', loss = "categorical_cross_entropy", metrics = ['accuracy'])
        self.model = model

    def get_model(self):
        '''
        Returns:
        --------
            self.model: the rnn being used
        '''
        return self.model

def train_test_RNN(data, X_train, Y_train, test_x, test_y):
    rnn = RNN_model(data)
    model = rnn.get_model()
    model.fit(X_train, Y_train, epochs = 10, batch_size = 32)
    test_loss, acc = model.evaluate(test_x, test_y)
