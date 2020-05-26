import numpy as np
import data_processing as data
import os
import cnn


#data_path = os.path.join('/Users/sunnyshah/Desktop/Spring 2020/CS 230/Project/3D_CNN', 'data')
#labels_path = os.path.join(data_path, 'labels.csv')


#df = data.Dataset(data_path, labels_path)
#num_classes = df.get_num_classes() #190
#img_size = df.get_image_size() #(min_frames, 125, 150, 3)

train_paths = ('x_train.npy', 'y_train.npy')
test_paths = ('x_test.npy', 'y_test.npy')
#x = np.load('x_train.npy')
#print(x.shape)
#print(num_classes, img_size)

img_size = (105, 125, 150, 3)
num_classes = 191
cnn_model = cnn.ConvNet(img_size, num_classes) # create a 3D-CNN
cnn_model.summary() # print summary
cnn_model.train(train_paths, test_paths) # trains model with epochs = 50, batch_size = 16
