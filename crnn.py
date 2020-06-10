import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization, LSTM, Reshape
from tensorflow.keras import optimizers, losses, regularizers
from tensorflow.keras import models
from tensorflow.keras.callbacks import TensorBoard
import datetime

class CRNN:
    def __init__(self, img_shape, num_classes, model_num):
        self.img_shape = img_shape
        self.num_classes = num_classes
        model = None

        if model_num == 1: # just regular network
            model = self.exp1()

        elif model_num == 2: # with L2-regularization
            model = self.exp2()

        elif model_num == 3: # with L1-regularization
            model = self.exp3()

        elif model_num == 4: # with dropout
            model = self.exp4()

        elif model_num == 5: # with L2-regularization + dropout
            model = self.exp5()

        elif model_num == 6: # with L1-regularization + dropout
            model = self.exp6()

        self.model = model

    def summary(self):
        self.model.summary()

    ############################################################################

    def exp1(self):
        """
        build CNN-RNN model
        """
        img_input = Input(shape = self.img_shape)
        x = Conv2D(filters = 64, kernel_size = 3, padding = 'same')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 128, kernel_size = 3, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same')(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = (2, 2), strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Reshape((-1, 512))(x)

        #RNN
        x = LSTM(units = 64, return_sequences=True)(x)
        x = LSTM(units = 64, return_sequences=False)(x)

        x = Dense(units = num_classes)(x)
        x = Activation('softmax')(x)

        return keras.Model(inputs = img_input, outputs = x, name='CRNN')

    ############################################################################

    def exp2(self):
        """
        build CNN-RNN model
        """
        img_input = Input(shape = self.img_shape)
        x = Conv2D(filters = 64, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 128, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = (2, 2), strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 2, kernel_regularizer = regularizers.l2(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Reshape((-1, 512))(x)

        #RNN
        x = LSTM(units = 64, return_sequences = True, kernel_regularizer = regularizers.l2(0.1))(x)
        x = LSTM(units = 64, return_sequences = False, kernel_regularizer = regularizers.l2(0.1))(x)

        x = Dense(units = num_classes)(x)
        x = Activation('softmax')(x)

        return keras.Model(inputs = img_input, outputs = x, name='CRNN')

    ############################################################################

    def exp3(self):
        """
        build CNN-RNN model
        """
        img_input = Input(shape = self.img_shape)
        x = Conv2D(filters = 64, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 128, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = (2, 2), strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 2, kernel_regularizer = regularizers.l1(0.1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Reshape((-1, 512))(x)

        #RNN
        x = LSTM(units = 64, return_sequences=True, kernel_regularizer = regularizers.l1(0.1))(x)
        x = LSTM(units = 64, return_sequences=False, kernel_regularizer = regularizers.l1(0.1))(x)

        x = Dense(units = num_classes)(x)
        x = Activation('softmax')(x)

        return keras.Model(inputs = img_input, outputs = x, name='CRNN')

    ############################################################################

    def exp4(self):
        """
        build CNN-RNN model
        """
        img_input = Input(shape = self.img_shape)
        x = Conv2D(filters = 64, kernel_size = 3, padding = 'same')(img_input)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 128, kernel_size = 3, padding = 'same')(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same')(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same')(x)
        x = Dropout(rate = 0.4)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same')(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same')(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = (2, 2), strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 2)(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Reshape((-1, 512))(x)

        #RNN
        x = LSTM(units = 64, return_sequences=True)(x)
        x = LSTM(units = 64, return_sequences=False)(x)

        x = Dense(units = num_classes)(x)
        x = Activation('softmax')(x)

        return keras.Model(inputs = img_input, outputs = x, name='CRNN')

    ############################################################################

    def exp5(self):
        """
        build CNN-RNN model
        """
        img_input = Input(shape = self.img_shape)
        x = Conv2D(filters = 64, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(img_input)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 128, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l2(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = (2, 2), strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 2, kernel_regularizer = regularizers.l2(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Reshape((-1, 512))(x)

        #RNN
        x = LSTM(units = 64, return_sequences = True, kernel_regularizer = regularizers.l2(0.1))(x)
        x = LSTM(units = 64, return_sequences = False, kernel_regularizer = regularizers.l2(0.1))(x)

        x = Dense(units = num_classes)(x)
        x = Activation('softmax')(x)

        return keras.Model(inputs = img_input, outputs = x, name='CRNN')

    ############################################################################

    def exp6(self):
        """
        build CNN-RNN model
        """
        img_input = Input(shape = self.img_shape)

        x = Conv2D(filters = 64, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(img_input)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 128, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, padding = 'same')(x)

        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters = 256, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = 2, strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters = 512, kernel_size = 3, padding = 'same', kernel_regularizer = regularizers.l1(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size = (2, 2), strides = (2, 1), padding = 'same')(x)

        x = Conv2D(filters = 512, kernel_size = 2, kernel_regularizer = regularizers.l1(0.1))(x)
        x = Dropout(rate = 0.4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Reshape((-1, 512))(x)

        #RNN
        x = LSTM(units = 64, return_sequences=True, kernel_regularizer = regularizers.l1(0.1))(x)
        x = LSTM(units = 64, return_sequences=False, kernel_regularizer = regularizers.l1(0.1))(x)

        x = Dense(units = num_classes)(x)
        x = Activation('softmax')(x)

        return keras.Model(inputs = img_input, outputs = x, name='CRNN')


    def train(self, train_paths, test_paths, num_epochs = 7, batch = 1024, lr = 0.0001):
        x_train, y_train = np.load(train_paths[0]), np.load(train_paths[1])
        x_test, y_test = np.load(test_paths[0]), np.load(test_paths[1])

        #Tensorboard
        # logs = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard = TensorBoard(log_dir = logs, histogram_freq = 1)
        
        opt = optimizers.Adam(learning_rate = lr)
        self.model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = self.model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch)
        # self.history = self.model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch, callbacks = [tensorboard])

        print('-' * 35)
        print('-' * 35)
        print("Finished training....\nNow Evaluating:")

        train_eval = self.model.evaluate(x_train, y_train.T, verbose = 0)
        print("Train evaluation:\n\tLoss = {}\n\tAccuracy = {:.2f}%".format(train_eval[0], train_eval[1] * 100))

        test_eval = self.model.evaluate(x_test, y_test.T, verbose = 0)
        print("Test evaluation:\n\tLoss = {}\n\tAccuracy = {}%".format(test_eval[0], test_eval[1] * 100))

        plt.plot(history.history['loss'], label = 'Loss')
        plt.plot(history.history['val_loss'], label = 'Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        file_name = exp_name + '_loss.png'
        plt.savefig(file_name)
        plt.clf()

        plt.plot(history.history['accuracy'], label = 'Accuracy')
        plt.plot(history.history['val_accuracy'], label = 'Val accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        file_name = exp_name + '_accuracy.png'
        plt.savefig(exp_name)
        plt.clf()

    ############################################################################
    ############################################################################
    ############################################################################

def plot(opt, rates, losses, accuracies):
    x = np.arange(len(accuracies[0])) + 1

    for i in range(len(rates)):
        plt.plot(x, accuracies[i], label = f'$r$ = {rates[i]:.2e}')

    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    file_name = 'accuracy_' + opt + '.png'
    plt.savefig(file_name)
    plt.clf()

    for i in range(len(rates)):
        plt.plot(x, losses[i], label = f'$r$ = {rates[i]:.2e}')

    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    file_name = 'loss_' + opt + '.png'
    plt.savefig(file_name)
    plt.clf()


def vary_learning_rate(train_path, test_path, img_size, num_classes, optimizer = 'Adam'):
    x_train, y_train = np.load(train_path[0]), np.load(train_path[1])
    x_test, y_test = np.load(test_path[0]), np.load(test_path[1])

    rates = np.logspace(-6, -2, 5)
    accuracies = []
    losses = []

    for i in range(len(rates)):
        print(f"Learning rate = {rates[i]:.3e}")
        crnn = ConvNet(img_size, num_classes, model_num = 1)
        model = crnn.get_model()

        if optimizer == 'Adam':
            opt = optimizers.Adam(learning_rate = rates[i]) 

        else:
            opt = optimizers.SGD(learning_rate = rates[i])

        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = model.fit(x_train, y_train.T, epochs = 7, batch_size = 9)

        accuracies.append(hist.history['accuracy'])
        losses.append(hist.history['loss'])
        print('-' * 35)
        print('-' * 35)

    plot(optimizer, rates, losses, accuracies)
