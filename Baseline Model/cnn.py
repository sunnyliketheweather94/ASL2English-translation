# used for setting up the CNN
# based on the code from https://github.com/harvitronix/five-video-classification-methods/blob/master/extractor.py
import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import cv2


class Inception_Model:
    def __init__(self):
        '''
        we set up the InceptionV3 model but end it at the final pool layer
        so that the features can be extracted from here and then 
        be inputted into the upcoming LSTM
        '''
        base_model = InceptionV3(weights = 'imagenet', include_top = True)
        self.model = models.Model(inputs = base_model.input,
                                  outputs = base_model.get_layer('avg_pool').output)

    def get_model(self):
        '''
        Returns:
        --------
            self.model: the convolutional network model being used

        we use the pre-trained InceptionV3 network that has already been trained 
        on a million or so images, called the ImageNet
        '''
        return self.model


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