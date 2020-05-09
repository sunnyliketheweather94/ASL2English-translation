import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import data_processing
import matplotlib.pyplot as plt 

class RNN_model:
    def __init__(self, data, input_size, frames):
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
        model = models.Sequential()
        # inputs = 
        model.add(keras.Input(shape = [input_size, frames]))
        model.add(layers.LSTM(64, dropout = 0.2, return_sequences = True))
        model.add(layers.LSTM(64, dropout = 0.2, return_sequences = True))
        model.add(layers.LSTM(64, dropout = 0.2))
        model.add(layers.Dense(data.get_num_classes(), activation = 'softmax'))
        model.compile(optimizer = 'Adam', loss = "categorical_crossentropy", metrics = ['accuracy', 'categorical_accuracy'])
        self.model = model

    def get_model(self):
        '''
        Returns:
        --------
            self.model: the rnn being used
        '''
        return self.model

def train_test_RNN(data, xtr_path, ytr_path, xts_path, yts_path):
    '''
    we obain the train and test data, set up the RNN 
    we then train the RNN on the train data and evaluate it on the test data
    we then plot the accuracies and losses for the train/test data over epoch
    '''
    x_train = np.load(xtr_path)
    y_train = np.load(ytr_path)
    x_test = np.load(xts_path)
    y_test = np.load(yts_path)

    rnn = RNN_model(data, 2048, x_train.shape[2]) # x_train.shape[2] = max_num_frames
    model = rnn.get_model()

    print(model.summary())

    history = model.fit(x_train, y_train, epochs = 10, batch_size = 32)
    x = model.evaluate(x_test, y_test)

    print(x)

    # plot the accuracy over epochs
    plt.plot(history.history['categorical_accuracy'])
    plt.title('Model categorical accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    # plot the categorical accuracy over epochs
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    # plot the losses
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


