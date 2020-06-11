import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
import glob
from sklearn.utils import shuffle
import tensorflow.keras.preprocessing as tf_pre 
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical

pd.set_option('display.max_colwidth', 100)

class Dataset:
    def __init__(self, data_path, labels_path):
        self.total_frames = 0
        self.train = {}
        self.test = {}

        df = pd.read_csv(labels_path)

        FramesPerVideoPath = os.path.join(data_path, 'FramesPerVideo.csv')
        FramesPerVideo = pd.read_csv(FramesPerVideoPath)

        df['FramesPerVideo'] = FramesPerVideo['Frames']
        df['FramesPerVideo'] = df['FramesPerVideo'].astype(str).astype(int)
        df['Label'] = df['Label'].astype("str")

        for item in df['FramesPerVideo'].values:
            self.total_frames += item

        paths = [os.path.join(folder, s) for folder, subfolder, _ \
                 in os.walk(data_path) for s in sorted(subfolder) if "video" in s]

        # stores all the paths to frames for a given video
        frame_paths = []
        for path in paths:
            video_number = int(path.split('/')[-1][6:])

            path_list = []
            for filepath in glob.glob(path + "/*.jpg"):
                path_list.append(filepath)
        
            path_list.sort(key = lambda x : x.split('_')[1])
            frame_paths.append(path_list)

        df['FramePaths'] = frame_paths

        self.max_frames = np.max([i for i in df['FramesPerVideo']])
        self.min_frames = np.min([i for i in df['FramesPerVideo']])

        shuffled = shuffle(df)
        self.data = shuffled

        # 80% train, 20% test
        self.train = shuffled[:-int(len(df) * 0.2)]
        self.test = shuffled[-int(len(df) * 0.2):]

        self.total_train_frames = self.train['FramesPerVideo'].sum
        self.total_test_frames = self.test['FramesPerVideo'].sum

        print("Train data: {}".format(len(self.train)))
        print("Test data: {}".format(len(self.test)))

        # create a dictionary of all the words in the labels
        # the value for each word is going to be its index in the oneHot encoding (map each word to an index)
        label_dict = {}
        i = 0
        for word in shuffled['Label']:
            # if the word doesn't exist in the dictionary already..
            # add it and increment the counter
            if word not in label_dict.keys():
                label_dict[word] = i
                i += 1

        self.Words2Int = label_dict
        self.Int2Words = {c : i for i, c in self.Words2Int.items()}
        self.num_classes = len(self.Words2Int)

        self.img_size = (50, 60, 3)


    def get_frame_paths(self, video_number):
        row = self.data.loc[self.data['Video'] == video_number]
        paths = [i for p in row['FramePaths'] for i in p]

        return paths

    def get_image_size(self):
        return self.img_size

    def get_num_frames(self, video_number):
        row = self.data.loc[self.data['Video'] == video_number]
        return np.squeeze(self.data['FramesPerVideo'].to_numpy())

    def get_label(self, video_number):
        '''
        obtain the true label for a given video
        '''
        # get the row corresponding to this video
        data_ = self.data.loc[self.data['Video'] == video_number]
        label = (data_['Label']).to_string(index=False)

        return label[1:]

    def print_dictionary(self):
        '''
        print the contents of the dictionary
        '''
        for k, v in self.Words2Int.items():
            print("{}: {}".format(v, k))

    def get_num_classes(self):
        '''
        obtain the maximum number of words available in the dictionary
        '''
        return self.num_classes

    def get_max_frames(self):
        '''
        obtain the maximum number of frames amongst all the videos in the dataset
        '''
        return self.max_frames

    def get_min_frames(self):
        '''
        obtain the maximum number of frames amongst all the videos in the dataset
        '''
        return self.min_frames

    def get_oneHotIndex(self, video_number):
        label = self.get_label(video_number)

        return self.Words2Int[label]

    # 2-D matrix representation of frame image
    def get_frame_matrix(self, frame_path):
        img = cv2.imread(frame_path)
        img = cv2.resize(img, (60, 50))
        x = np.array(img)
        x = x/255
        x = np.expand_dims(x, axis=0)
        return x

    # 3-D volume representation of video (including time dimension, which equals # of frames in video)
    def get_video_volume(self, video_number, counter):
        frame_paths = self.get_frame_paths(video_number)

        volume = np.zeros((self.max_frames, 50, 60, 3))

        for i, p in enumerate(frame_paths):
            volume[i, :, :, :] = self.get_frame_matrix(p)

        print("{}. Size for video {} is {}.".format(counter, video_number, volume.shape))

        return volume

    def get_3Dcnn_data(self, type_):
        if type_ == 'train':
            data = self.train
        else:
            data = self.test

        video_numbers = data['Video']

        # create x_train - one training example per video
        counter = 1
        x = np.zeros((len(data), self.max_frames, 50, 60, 3))
        for i, vid_num in enumerate(video_numbers):
            counter += 1
            x[i, :, :, :, :] = self.get_video_volume(vid_num, counter)
            
        print("the total size of x is {}".format(x.shape))

        # create y_train - one index per video
        y = np.zeros((self.num_classes, len(data)))
        print("y shape: ", y.shape)

        for i, vid_num in enumerate(video_numbers):
            idx = self.get_oneHotIndex(vid_num)
            y[idx, i] = 1

        print("the total size of y is {}".format(y.shape))

        return x, y

    def get_crnn_data(self, type_):
        frames_ = 0

        if type_ == 'train':
            data = self.train
        else:
            data = self.test

        video_numbers = data['Video']

        '''Create x_train(no padding)'''
        i = 0
        #count frames per video to initialize x_train/test shape
        for vid_num in video_numbers:
            frame_paths = self.get_frame_paths(vid_num)
            for path in frame_paths:
                i += 1
        x = np.zeros((i, 50, 60, 3))
        y = np.zeros((i, self.num_classes))
        i = 0
        #create x_train/test (one training example per frame) and y_train (same label for all frames in a particular video)
        for vid_num in video_numbers:
            frame_paths = self.get_frame_paths(vid_num)
            for path in frame_paths:
                x[i, :, :, :] = self.get_frame_matrix(path)
                idx = self.get_oneHotIndex(vid_num)
                y[i, idx] = 1
                i += 1
            print("Done with video:", vid_num)
        print("the total size of x is {}".format(x.shape))
        print("the total size of y is {}".format(y.shape))

        return x, y




