import numpy as np 
import matplotlib.pyplot as pyplot
import os
import cv2

import cnn
import rnn
import data_processing

# for teresa's local computer
#labels_path = os.path.join('/Users/teresanoyola/Desktop/CS230/Project/ASL-English-translation/Baseline Model', 'labels.csv') # CHANGE THIS
#data_path = '/Users/teresanoyola/Desktop/CS230/Project/Processed Data/' # CHANGE THIS

# for sunny's local computer
labels_path = os.path.join('../../Data', 'labels.csv') # CHANGE THIS
data_path = '../../Data/Processed Data/' # CHANGE THIS

# for georgia



# Teresa's Google Drive
# labels_path = '/content/drive/My Drive/CS230/Data/labels.csv'
# data_path = '/content/drive/My Drive/CS230/Data/Processed Data'

data = data_processing.Dataset(data_path, labels_path)
#print(data.get_max_frames())
data.get_data_for_CNN()

#model = cnn.Inception_Model(retrain = True)
#model.model_retrain('x_train.npy', 'y_train.npy')
#rnn.train_test_RNN(data, 'x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy')