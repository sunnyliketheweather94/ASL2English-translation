import numpy as np 
import matplotlib.pyplot as pyplot
import os
import cv2

#import cnn
#import rnn
#import data_processing

#labels_path = os.path.join('/Users/teresanoyola/Desktop/CS230/Project/ASL-English-translation/Baseline Model', 'labels.csv') # CHANGE THIS
#data_path = '/Users/teresanoyola/Desktop/CS230/Project/Processed Data/' # CHANGE THIS
#Google Drive
labels_path = '/content/drive/My Drive/CS230/Data/labels.csv'
data_path = '/content/drive/My Drive/CS230/Data/Processed Data'

data = Dataset(data_path, labels_path)
#print(data.get_max_frames())
data.get_data_for_CNN()

#model = cnn.Inception_Model(retrain = True)
#model.model_retrain('x_train.npy', 'y_train.npy')
#rnn.train_test_RNN(data, 'x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy')