import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Activation, Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import optimizers, losses
from tensorflow.keras import models

class CRNN:
    def __init__(self, img_shape, num_classes, model_num):
        self.img_shape = img_shape
        self.num_classes = num_classes

        if model_num == 1:
            model = self.get_model1(img_shape, num_classes)

        # elif model_num == 2:
        #     model = self.get_model2()
        #
        # elif model_num == 3:
        #     model = self.get_model3()


        self.model = model

    def summary(self):
        self.model.summary()

    def get_model1(self, input_shape, num_classes):
        """
        build CNN-RNN model
        """
        def vgg_style(input_tensor):
            """
            The original feature extraction structure from CRNN paper.
            Related paper: https://ieeexplore.ieee.org/abstract/document/7801919
            """
            x = layers.Conv2D(
                filters=64,
                kernel_size=3,
                padding='same',
                activation='relu')(input_tensor)
            x = layers.MaxPool2D(pool_size=2, padding='same')(x)

            x = layers.Conv2D(
                filters=128,
                kernel_size=3,
                padding='same',
                activation='relu')(x)
            x = layers.MaxPool2D(pool_size=2, padding='same')(x)

            x = layers.Conv2D(filters=256, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters=256, kernel_size=3, padding='same',
                              activation='relu')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1),
                                 padding='same')(x)

            x = layers.Conv2D(filters=512, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters=512, kernel_size=3, padding='same',
                              activation='relu')(x)
            x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1),
                                 padding='same')(x)

            x = layers.Conv2D(filters=512, kernel_size=2)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            return x

        img_input = keras.Input(shape=self.img_shape)
        #CNN
        x = vgg_style(img_input)
        x = layers.Reshape((-1, 512))(x)
        #x = layers.Flatten()(x)


        #RNN
        x = layers.LSTM(units=256, return_sequences=True))(x)
        x = layers.LSTM(units=256, return_sequences=False))(x)
        x = layers.Dense(units=num_classes)(x)
        x = layers.Activation('softmax')(x)
        return keras.Model(inputs=img_input, outputs=x, name='CRNN')


    def train(self, train_paths, test_paths, num_epochs=3, batch=512, lr=0.0001):
        xtr_path, ytr_path = train_paths
        xts_path, yts_path = test_paths

        x_train = np.load(xtr_path)
        y_train = np.load(ytr_path)

        x_test = np.load(xts_path)
        y_test = np.load(yts_path)

        opt = optimizers.Adam(learning_rate=lr, beta_1=0.95, beta_2=0.98)
        # opt = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch)

        train_eval = self.model.evaluate(x_test, y_test)
        print("Test evaluation:\nLoss = {}\nAccuracy = {}".format(train_eval[0], train_eval[1]))

        #test_eval = self.model.evaluate(x_test, y_test.T)
        #print("Test evaluation:\nLoss = {}\nAccuracy = {}".format(test_eval[0], test_eval[1]))


        self.plot_accuracy()
        self.plot_loss()

        return train_eval

    def plot_accuracy(self, save=True):
        hist = self.history

        # plot the accuracy over epochs
        plt.plot(hist.history['accuracy'])
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        if not save:
            plt.show()

        else:
            plt.savefig('accuracy.png')

    def plot_loss(self, save=True):
        hist = self.history

        # plot the loss over epochs
        plt.plot(hist.history['loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        if not save:
            plt.show()

        else:
            plt.savefig('loss.png')

def plot(opt, rates, losses, accuracies):
    for i in range(len(rates)):
        plt.plot(rates[i], accuracies[i], label = f'$r$ = {rates[i]:.3f}')

    plt.xlabel('Learning rates')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy versus Learning Rate')
    file_name = 'accuracy_' + opt + '.png'
    plt.savefig(file_name)

    for i in range(len(rates)):
        plt.plot(rates[i], losses[i], label = f'$r$ = {rates[i]:.3f}')

    plt.xlabel('Learning rates')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss versus Learning Rate')
    file_name = 'loss_' + opt + '.png'
    plt.savefig(file_name)
