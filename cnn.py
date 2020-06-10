import numpy as np
import matplotlib.pyplot as plt 

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Activation, Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras import optimizers, losses, regularizers
from tensorflow.keras import models


class ConvNet:
    def __init__(self, shape, num_classes, model_num):
        self.img_shape = shape
        self.num_classes = num_classes
        model = None

        if model_num == 1:
            model = self.exp1()

        elif model_num == 2:
            model = self.exp2()

        elif model_num == 3:
            model = self.exp3()

        elif model_num == 4:
            model = self.exp4()

        elif model_num == 5:
            model = self.exp5()

        elif model_num == 6:
            model = self.exp6()

        self.model = model

    def summary(self):
        self.model.summary()

    def get_model(self):
        return self.model

    def exp1(self): # plain old regular network
        model = models.Sequential()

        model.add(Conv3D(16, kernel_size = (3, 3, 3), input_shape = self.img_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Conv3D(16, kernel_size = (3, 3, 3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Flatten())
        model.add(Dense(250, activation = 'relu'))
        model.add(Dense(self.num_classes, activation = 'softmax'))

        return model

    def exp2(self): # just L2-regularization
        model = models.Sequential()

        model.add(Conv3D(16, kernel_size = (3, 3, 3), input_shape = self.img_shape, kernel_regularizer = regularizers.l2(0.1)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Conv3D(32, kernel_size = (3, 3, 3), kernel_regularizer = regularizers.l2(0.1)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Flatten())
        model.add(Dense(250, activation = 'relu'))
        model.add(Dense(self.num_classes, activation = 'softmax'))

        return model

    def exp3(self): # just L1-regularization
        model = models.Sequential()

        model.add(Conv3D(16, kernel_size = (3, 3, 3), input_shape = self.img_shape, kernel_regularizer = regularizers.l1(0.1)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Conv3D(32, kernel_size = (3, 3, 3), kernel_regularizer = regularizers.l1(0.1)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Flatten())
        model.add(Dense(250, activation = 'relu'))
        model.add(Dense(self.num_classes, activation = 'softmax'))

        return model

    def exp4(self): # just dropout
        model = models.Sequential()

        model.add(Conv3D(16, kernel_size = (3, 3, 3), input_shape = self.img_shape))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Conv3D(32, kernel_size = (3, 3, 3)))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Flatten())
        model.add(Dense(250, activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation = 'softmax'))

        return model

    def exp5(self): # dropout + L2-regularization
        model = models.Sequential()

        model.add(Conv3D(16, kernel_size = (3, 3, 3), input_shape = self.img_shape, kernel_regularizer = regularizers.l2(0.1)))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Conv3D(32, kernel_size = (3, 3, 3), kernel_regularizer = regularizers.l2(0.1)))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Flatten())
        model.add(Dense(250, activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation = 'softmax'))

        return model

    def exp6(self): # dropout + L1-regularization
        model = models.Sequential()

        model.add(Conv3D(16, kernel_size = (3, 3, 3), input_shape = self.img_shape, kernel_regularizer = regularizers.l1(0.1)))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Conv3D(32, kernel_size = (3, 3, 3), kernel_regularizer = regularizers.l1(0.1)))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

        model.add(Flatten())
        model.add(Dense(250, activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation = 'softmax'))

        return model


    def train(self, train_paths, test_paths, exp_name, num_epochs = 7, batch = 4, lr = 0.0001):
        '''
        exp_name = experiment number (check __init__ for the exp num)
        e.g. if we're running on exp 5 for regular network (model 1), then exp_num = 15

        saves losses/accuracies under name: "15_loss.png" and "15_accuracy.png" if exp_num = 15
        '''
        x_train = np.load(train_paths[0])
        y_train = np.load(train_paths[1])

        x_test = np.load(test_paths[0])
        y_test = np.load(test_paths[1])

        # logs = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard = TensorBoard(log_dir = logs, histogram_freq = 1)

        opt = optimizers.Adam(learning_rate = lr)
        self.model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history = self.model.fit(x_train, y_train.T, epochs = num_epochs, batch_size = batch, validation_split = 0.2)
        # history = self.model.fit(x_train, y_train.T, epochs = num_epochs, batch_size = batch, validation_split = 0.2, callbacks = [tensorboard])
        
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

    # def get_deeper_CNN(self):
    #     model = models.Sequential()

    #     model.add(Conv3D(16, kernel_size = (3, 3, 3), input_shape = self.img_shape, activation = 'relu',  kernel_regularizer = regularizers.l2(0.01)))
    #     # model.add(Conv3D(16, kernel_size = (3, 3, 3), input_shape = self.img_shape, activation = 'relu'))
    #     # model.add(Dropout(0.4))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

    #     model.add(Conv3D(16, kernel_size = (3, 3, 3), activation = 'relu', kernel_regularizer = regularizers.l2(.1)))
    #     # model.add(Conv3D(16, kernel_size = (3, 3, 3), activation = 'relu'))
    #     # model.add(Dropout(0.4))
    #     # model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

    #     model.add(Conv3D(32, kernel_size = (3, 3, 3), activation = 'relu', kernel_regularizer = regularizers.l2(.1)))
    #     # model.add(Conv3D(16, kernel_size = (3, 3, 3), activation = 'relu'))
    #     # model.add(Dropout(0.4))
    #     # model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

    #     model.add(Conv3D(32, kernel_size = (3, 3, 3), activation = 'relu', kernel_regularizer = regularizers.l2(.1)))
    #     # model.add(Conv3D(16, kernel_size = (3, 3, 3), activation = 'relu'))
    #     # model.add(Dropout(0.4))
    #     # model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

    #     model.add(Flatten())
    #     model.add(Dense(1000, activation = 'sigmoid'))
    #     model.add(Dropout(0.4))
    #     model.add(Dense(500, activation = 'sigmoid'))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(self.num_classes, activation = 'softmax'))

    #     return model



    # def get_wider_CNN(self):
    #     model = models.Sequential()

    #     model.add(Conv3D(16, kernel_size = (3, 3, 3), input_shape = self.img_shape))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

    #     model.add(Conv3D(128, kernel_size = (3, 3, 3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

    #     model.add(Conv3D(32, kernel_size = (3, 3, 3)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'valid'))

    #     model.add(Flatten())
    #     model.add(Dense(1024, activation = 'sigmoid'))
    #     model.add(Dense(self.num_classes, activation = 'softmax'))

    #     return model

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
        cnn_model = ConvNet(img_size, num_classes, model_num = 1)
        model = cnn_model.get_model()

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



