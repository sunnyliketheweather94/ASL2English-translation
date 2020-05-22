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
data.get_data_for_CNN()
#model = cnn.Inception_Model(retrain = True)
#model.model_retrain('x_train.npy', 'y_train.npy')
#rnn.train_test_RNN(data, 'x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy')

'''#remove padding
x_train = np.load('x_train.npy') #n_vids x 2048 x max_frames
n_vids, _ , max_frames = x_train.shape
for i in range(n_vids): #loop over num_vids
    matrix = x_train[i] #2048 x max_frames
    for j in range(max_frames):
        if np.array_equal(matrix[:,j], np.zeros((2048,)):
            frames = j
    temp = matrix[:,:frames'''