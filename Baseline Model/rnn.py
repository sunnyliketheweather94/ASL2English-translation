import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import data_processing
import matplotlib.pyplot as plt 

class RNN_model:
    def __init__(self, frames, input_size):
        '''
        set up the RNN model as outlined in Bantupalli and Xie (2018)
        ---------------------
        input to the RNN is a matrix of size (num_videos, max_frames, 2048)
            2048 is the length of the output from the CNN

        Layers in the RNN:
        ------------------
        LSTM -> LSTM -> LSTM -> Dense

        Other components of the model:
        -------------------
        optimizer = Adam
        loss function = categorical cross entropy
        metrics = accuracy
        '''
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

def train_test_RNN(xtr_path, ytr_path, xts_path, yts_path):
    '''
    we obain the train and test data, set up the RNN 
    we then train the RNN on the train data and evaluate it on the test data
    we then plot the accuracies and losses for the train/test data over epoch
    '''
    x_train = np.load(xtr_path)
    y_train = np.load(ytr_path)
    x_test = np.load(xts_path)
    y_test = np.load(yts_path)

    rnn = RNN_model(x_train.shape[2], 2048)
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


