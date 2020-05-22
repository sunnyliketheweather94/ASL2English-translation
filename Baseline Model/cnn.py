# used for setting up the CNN
# based on the code from https://github.com/harvitronix/five-video-classification-methods/blob/master/extractor.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt


class Inception_Model:
    def __init__(self, retrain, weights = 'imagenet'):
        '''
        we set up the InceptionV3 model but end it at the final pool layer
        so that the features can be extracted from here and then
        be inputted into the upcoming LSTM
        '''
        self.retrain = retrain #indicator
        if retrain:
            self.model = self.setup_retrain_model()

        else:
            self.model = self.setup_model(weights)

    def setup_model(self, weights):
        base_model = InceptionV3(weights = weights, include_top = False)
        model = models.Model(inputs = base_model.input,
                                  outputs = base_model.get_layer('avg_pool').output)
        return model

    def setup_retrain_model(self):
        # CNN (frozen + trainable layers)
        # trainable layers will be trained and weights saved
        base_model = InceptionV3(weights = 'imagenet', include_top = True)
        model = models.Model(inputs=base_model.input,
                             outputs=base_model.get_layer('avg_pool').output)

        model.trainable = True

        fine_tune_at = 250 # Fine-tune from 'average_pooling2d_7' onwards
        for layer in model.layers[:fine_tune_at]: # Freeze all the layers before the `fine_tune_at` layer
            layer.trainable = False

        model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=['categorical_accuracy'])
        return model

    def get_model(self):
        '''
        Returns:
        --------
            self.model: the convolutional network model being used

        we use the pre-trained InceptionV3 network that has already been trained 
        on a million or so images, called the ImageNet
        '''
        return self.model


    def model_retrain(self, xtr_path, ytr_path, xtst_path, ytst_path):
        # load x_train, y_train, x_test, y_test
        # input each column of the matrices (for each video) into the input layer of model_retrain
        # use the y_train as training labels
        # use the categorical_crossentropy function

        assert(self.retrain == True)
        x_train = np.load(xtr_path)
        y_train = np.load(ytr_path)
        x_test = np.load(xtst_path)
        y_test = np.load(ytst_path)

        hist = self.model.fit(x_train, y_train, epochs = 30, batch_size = 64)
        x = self.model.evaluate(x_test, y_test)

        print("Test evaluation:\nLoss = {}\nCategorical Accuracy = {}".format(x[0], x[1]))

        # plot the accuracy over epochs
        plt.plot(hist.history['categorical_accuracy'])
        plt.title('Model categorical accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

        # plot the loss over epochs
        plt.plot(hist.history['loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        self.model.save_weights('new_weights.h5py')


    def extract_features(self, img_path):
        '''
        We will feed the features obtained from the final pool layer
        into the LSTM.
        
        Arguments:
        ----------
            img_path: the path to the image whose features need to be extracted
        
        Returns:
        --------
            features[0]: the features of the image obtained from 
            the final pool layer of the network
        '''
        img = cv2.imread(img_path)
        img = cv2.resize(img, (299, 299))
        x = np.array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)

        # get the prediction
        features = self.model.predict(x)

        return features[0]