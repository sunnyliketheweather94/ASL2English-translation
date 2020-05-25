import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 
import os
from tensorflow.keras.layers import Activation, Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import optimizers, losses
from tensorflow.keras import models

class ConvNet:
    def __init__(self, shape, num_classes):
        model = models.Sequential()

        # frames, height, width, channels = shape
        # shape = (frames, height, width, channels)

        model.add(Conv3D(32, kernel_size = (3, 3, 3), input_shape = shape))
        model.add(Activation('relu'))

        # model.add(Conv3D(32, kernel_size = (3, 3, 3)))
        # model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size = (3, 3, 3)))

        model.add(Conv3D(64, kernel_size = (3, 3, 3)))
        model.add(Activation('relu'))
        # model.add(Dropout(0.20))

        # model.add(Conv3D(64, kernel_size = (3, 3, 3)))
        # model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(1024, activation = 'sigmoid'))
        # model.add(Dense(512, activation = 'sigmoid'))
        model.add(Dense(num_classes, activation = 'softmax'))

        model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
        self.model = model


    def summary(self):
        print(self.model.summary)

    def get_model(self):
        return self.model

    def train(self, train_paths, test_paths, num_epochs = 50, batch = 16):
        xtr_path, ytr_path = train_paths
        xts_path, yts_path = test_paths

        x_train = np.load(xtr_path)
        y_train = np.load(ytr_path)

        x_test = np.load(xts_path)
        y_test = np.load(yts_path)

        hist = self.model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch)
        test_eval = self.model.evaluate(x_test, y_test)

        print("Test evaluation:\nLoss = {}\nCategorical Accuracy = {}".format(x[0], x[1]))

        self.history = hist 

        self.plot_accuracy()
        self.plot_loss()

    def plot_accuracy(self):
        hist = self.history 

        # plot the accuracy over epochs
        plt.plot(hist.history['categorical_accuracy'])
        plt.title('Model categorical accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def plot_loss(self):
        hist = self.history 

        # plot the loss over epochs
        plt.plot(hist.history['loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
