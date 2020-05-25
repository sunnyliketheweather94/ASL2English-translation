import numpy as np
import data_processing as data
import os


data_path = os.path.join('/Users/sunnyshah/Desktop/Spring 2020/CS 230/Project/3D_CNN', 'data')
labels_path = os.path.join(data_path, 'labels.csv')


df = data.Dataset(data_path, labels_path)

x_train, y_train = df.get_data('train')
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)

x_test, y_test = df.get_data('test')
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

