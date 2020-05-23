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
# data.get_data_for_CNN()

test_order = [157, 119, 100, 172, 168, 
         116, 69, 5, 149, 60, 73, 
         118, 175, 194, 165, 197,
         185, 158, 31, 177, 188, 112, 
         17, 178, 40]

train_order = [150, 129, 140, 43, 162,
               90, 25, 70, 21, 74, 93,
               23, 79, 148, 170, 19,
               186, 106, 111, 2, 196, 95,
               146, 84, 137, 88, 130, 68, 
               191, 58, 48, 115, 156, 144,
               14, 36, 72, 78, 97, 110, 96,
               193, 99, 53, 147, 160, 15, 47,
               182, 163, 127, 7, 77, 123, 59, 
               132, 46, 51, 138, 107, 9, 94,
               26, 108, 12, 141, 180, 81, 154,
               54, 44, 125, 18, 38, 92, 61, 173,
               27, 52, 181, 39, 29, 117, 45, 159,
               71, 75, 187, 49, 56, 136, 166, 189, 
               183, 1, 28, 6, 32, 98, 113, 91, 174, 4,
               171, 82, 135, 122, 86, 143, 109, 3, 30, 
               85, 199, 13, 190, 50, 87, 105, 37, 8, 
               167, 133, 55, 121, 22, 102, 104, 16, 139,
               161, 114, 66, 34, 192, 145, 198, 131, 10, 103, 
               64, 11, 35, 134, 184, 164, 195, 83, 124, 57,
               101, 42, 128, 169, 176, 142, 67, 126, 80, 201,
               89, 155, 152, 24, 20, 41, 63, 200, 151, 33,
               62, 65, 120, 153]

test_order = ["matrix_" + str(p) + ".npy" for p in test_order]
train_order = ["matrix_" + str(p) + ".npy" for p in train_order]

data.concatenate_matrices('test', test_order)
data.concatenate_matrices('train', train_order)

#model = cnn.Inception_Model(retrain = True)
#model.model_retrain('x_train.npy', 'y_train.npy')
#rnn.train_test_RNN(data, 'x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy')