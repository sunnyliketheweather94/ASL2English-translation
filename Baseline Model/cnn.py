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
    def __init__(self, retrain):
        '''
        we set up the InceptionV3 model but end it at the final pool layer
        so that the features can be extracted from here and then 
        be inputted into the upcoming LSTM
        '''
        self.retrain = retrain 
        if retrain:
            self.model_1, self.model_2 = setup_retrain_model()

        else:
            self.model = setup_model()

    def setup_new_model(self):
        # CNN but with the pretrained + new weights layers
        base_model = InceptionV3(weights = 'imagenet', include_top = True)
        model = models.Model(inputs = base_model.input, )

    def setup_model(self):
        base_model = InceptionV3(weights = 'imagenet', include_top = True)
        model = models.Model(inputs = base_model.input,
                                  outputs = base_model.get_layer('avg_pool').output) 
                                  
        return model       

    def setup_retrain_model(self):
        # CNN (frozen + trainable layers)
        # trainable layers will be trained and weights saved
        base_model = InceptionV3(weights = 'imagenet', include_top = True)
        model_1 = models.Model(inputs = base_model.input,
                                  outputs = base_model.get_layer('avg_pool_7').output)

        model_2 = models.Model(inputs = base_model.get_layer('conv2d_70').input,
                                  outputs = base_model.get_layer('avg_pool').output)

        return model_1, model_2

    def get_model(self):
        '''
        Returns:
        --------
            self.model: the convolutional network model being used

        we use the pre-trained InceptionV3 network that has already been trained 
        on a million or so images, called the ImageNet
        '''
        return self.model


    def model_retrain(self):
        # load x_train, y_train, x_test, y_test
        # input each column of the matrices (for each video) into the input layer of model_2
        # use the y_train as training labels
        # use the categorical_crossentropy function
        # maybe look into the retraining of Inception-V3??

        # set up different vectors (each makes up a column of a matrix for a video)
        # input that vector with the corresponding label (for the same video)
        # ignore the padding!! when setting up the matrices x_train/x_test
        assert(self.retrain == True)
        x_train = np.load(xtr_path)
        y_train = np.load(ytr_path)

        matrix = x_train[0] # matrix for 1st video of training data (7, max_frames)

        self.model_2.fit(x_train, y_train)

        self.model_2.save_weights('new_weights.h5py')


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
        features = self.model_1.predict(x)

        return features[0]