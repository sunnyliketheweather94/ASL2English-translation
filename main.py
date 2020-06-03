import numpy as np
import data_processing as data
import os
import cnn


data_path = '/Users/teresanoyola/Desktop/CS230/Project/Data'
labels_path = os.path.join(data_path, 'labels.csv')


df = data.Dataset(data_path, labels_path)

x_train, y_train = df.get_data('train')

np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)

x_test, y_test = df.get_data('test')

np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)


num_classes = df.get_num_classes()
img_size = df.get_image_size()
print(num_classes)
print(img_size)



#x_train = np.load('/home/ubuntu/x_train.npy')
#print(x_train.shape)

#y_train = np.load('/home/ubuntu/y_train.npy')
#print(y_train.shape)

#train_paths = ('/home/ubuntu/x_train.npy', '/home/ubuntu/y_train.npy')
#test_paths = ('/home/ubuntu/x_test.npy', '/home/ubuntu/y_test.npy')


# print(num_classes, img_size)

#img_size = (105, 125, 150, 3)
#num_classes = 191
#cnn_model = cnn.ConvNet(img_size, num_classes, model_num = 1) # create a 3D-CNN
#cnn_model.summary() # print summary
#cnn_model.train(train_paths, test_paths) # trains model with epochs = 50, batch_size = 16
#cnn_model.vary_AdamLR(train_paths[0], train_paths[1], test_paths[0], test_paths[1])
#cnn_model.vary_SGDLR(train_paths[0], train_paths[1], test_paths[0], test_paths[1])
