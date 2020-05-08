import numpy as np 
import matplotlib.pyplot as pyplot
import os
import cv2

import cnn
# import rnn
import data_processing



labels_path = os.path.join('../../Data', 'labels.csv') ##CHANGE THIS
data_path = '../../Data/Processed Data/' #CHANGE THIS

data = data_processing.Dataset(data_path, labels_path)
train = data.get_testData()

model = cnn.Inception_Model()

# paths = train['frame_paths']
video_num = train['Video']

num = train['Video'][0]

paths = data.get_frame_paths(num)

f = model.extract_features(paths[0])
print(f)
print(len(f))