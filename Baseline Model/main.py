import numpy as np 
import matplotlib.pyplot as pyplot
import os
import cv2

import cnn
import rnn
import data_processing

labels_path = os.path.join('/Users/teresanoyola/Desktop/CS230/Project/ASL-English-translation/Baseline Model', 'labels.csv') # CHANGE THIS
data_path = '/Users/teresanoyola/Desktop/CS230/Project/Processed Data/' # CHANGE THIS

data = data_processing.Dataset(data_path, labels_path)
data.get_data_for_CNN(data.get_num_classes())
model = cnn.Inception_Model(True, n_classes = data.get_num_classes())
model.model_retrain('x_retrain.npy', 'y_retrain.npy')

rnn.train_test_RNN(data, 'x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy')