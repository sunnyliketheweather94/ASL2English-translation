import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import data_processing
import matplotlib.pyplot as plt 

class RNN_model:
    def __init__(self, data, frames, input_size):
        inputs = keras.Input(shape = [None, frames, input_size])
        model = layers.LSTM(64, dropout = 0.2, return_sequences = True)(inputs)
        model = layers.LSTM(64, dropout = 0.2, return_sequences = True)(model)
        model = layers.LSTM(64, dropout = 0.2, return_sequences = True)(model)
        model = layers.Dense(data.get_num_classes(), activation = 'softmax')
        model.compile(optimizer = 'Adam', loss = "categorical_cross_entropy", metrics = ['accuracy'])
        self.model = model

    def get_model(self):
        '''
        Returns:
        --------
            self.model: the rnn being used
        '''
        return self.model

def train_test_RNN(data):
    x_train, y_train, x_test, y_test = data.get_data_for_RNN()

    # print(x_tr.shape, x_ts.shape)

    rnn = RNN_model(data, data.get_max_frames(), 2048)
    model = rnn.get_model()

    history = model.fit(x_train, y_train, validation_split = 0.16, epochs = 10, batch_size = 32)
    test_loss, acc = model.evaluate(x_test, y_test)

    # plot the accuracy over epochs
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'])
    plt.show()

    # plot the losses
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train, validation'])
    plt.show()