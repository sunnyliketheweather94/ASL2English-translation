import numpy as np
import data_processing as data
import os
import cnn
import crnn
import utils


data_path = '/Users/teresanoyola/Desktop/CS230/Project/Data'
labels_path = os.path.join(data_path, 'labels.csv')

df = data.Dataset(data_path, labels_path)

x_train, y_train = df.get_crnn_data('train')
#print(x_train.shape)


np.save('/Users/teresanoyola/Desktop/CS230/Project/x_train_crnn.npy', x_train)
np.save('/Users/teresanoyola/Desktop/CS230/Project/y_train_crnn.npy', y_train)

#x_test, y_test = df.get_crnn_data('test')

#np.save('/Users/teresanoyola/Desktop/CS230/Project/x_test_crnn.npy', x_test)
#np.save('/Users/teresanoyola/Desktop/CS230/Project/y_test_crnn.npy', y_test)
#
#
num_classes = df.get_num_classes()
img_size = df.get_image_size()
print(num_classes)
print(img_size)
#
#
#
# #x_train = np.load('/home/ubuntu/x_train.npy')
# x_train = np.load('/Users/teresanoyola/Desktop/CS230/Project/x_train.npy')
# print(x_train.shape)
#
# y_train = np.load('/home/ubuntu/y_train.npy')
# print(y_train.shape)

train_paths = ('/Users/teresanoyola/Desktop/CS230/Project/x_train_crnn.npy', '/Users/teresanoyola/Desktop/CS230/Project/y_train_crnn.npy')
test_paths = ('/Users/teresanoyola/Desktop/CS230/Project/x_test_crnn.npy', '/Users/teresanoyola/Desktop/CS230/Project/y_test_crnn.npy')


# print(num_classes, img_size)

#img_size = (105, 125, 150, 3)
#num_classes = 191
#cnn_model = cnn.ConvNet(img_size, num_classes, model_num = 1) # create a 3D-CNN
#cnn_model.summary() # print summary
#cnn_model.train(train_paths, test_paths) # trains model with epochs = 50, batch_size = 16
#cnn_model.vary_AdamLR(train_paths[0], train_paths[1], test_paths[0], test_paths[1])
#cnn_model.vary_SGDLR(train_paths[0], train_paths[1], test_paths[0], test_paths[1])

'''CRNN'''
img_size = (50, 60, 3)
num_classes = 535
#print(num_classes, img_size)
#crnn_model = crnn.CRNN(img_size, num_classes, model_num = 1) # create a 3D-CNNcrnn_model.summary() # print summary
#crnn_model.train(train_paths, test_paths) # trains model with epochs = 50, batch_size = 16
# utils.vary_AdamLR(train_paths, test_paths)
# utils.vary_SGDLR(train_paths, test_paths)
