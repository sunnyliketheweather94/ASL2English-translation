import numpy as np 
import matplotlib.pyplot as pyplot
import os
import cv2

import cnn
import rnn
import data_processing

labels_path = os.path.join('../../Data', 'labels.csv') # CHANGE THIS
data_path = '../../Data/Processed Data/' # CHANGE THIS

data = data_processing.Dataset(data_path, labels_path)
x_train, y_train, x_test, y_test = data.get_data_for_CNN()

rnn.train_test_RNN(data, 'x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy')