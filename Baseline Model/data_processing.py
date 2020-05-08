# used for processing the data
import numpy as np
import pandas as pd 
import glob
import os
import cv2
from sklearn.utils import shuffle
import tensorflow.keras.preprocessing as pre 
from sklearn.model_selection import train_test_split as split 
import string

import cnn


# assume that we have labels already processed.
# labels are stored in 'labels.csv' where the first column is the video number
# and the second column is the corresponding label
class Dataset:
    '''
    Members: the following are for the train, dev and test data.
        num_frame: a list of the number of frames per video
                   the nth video's num_frame is num_frame[n - 1]
        file_names: a list of the file names of the videos
        frame_paths: a list of paths to frames for videos in the data
        labels: a list of the true labels
        df: the original dataframe containing all the information (in the original order) 

        they're all stored in a dictionary with the keys being train, dev
        and test, respectively. 

        for example, if you require the train_labels, you'd do:
                data = Dataset(path)
                test = data.getTestData() # a dictionary
                test_labels = test['labels']
    '''
    def __init__(self, data_path, label_path):
        '''
        data_path: the path to the processed data 
        label_path: the path to the processed label csv file
        '''
        self.train = {}
        self.test = {}
        self.img_shape = (231, 299, 3)

        df = pd.read_csv(label_path)

        # add the frames_per_video to the dataframe
        frames_per_vid_path = os.path.join(data_path, 'framesPerVid.csv')
        frames_per_video = pd.read_csv(frames_per_vid_path)
        df['FramesPerVideo'] = frames_per_video['FramesPerVideo']

        # essentially rename the Translations column as the Label column
        df['Label'] = df['Translations']
        df['Label'] = df['Label'].astype("string")
        print(df['Label'].head())
        df.drop('Translations', axis = 1, inplace = True)

        path_list = [os.path.join(folder, s) for folder, subfolder, _ \
                                in os.walk(data_path) for s in sorted(subfolder)]

        # obtain the paths for all the frames in the correct order
        # and then append them to a list for a given video
        # and then append that list to the frame_paths which will store
        # all the frame paths (in order) for all the videos
        frame_paths = []
        for path in path_list:
            video_num = int(path.split('/')[-1]) # get the video number of current video

            paths = []
            for filepath in glob.glob(path + '/*.jpg'):
                paths.append(filepath) # store the paths of all the frames in the subdirectory

            paths.sort(key = lambda x : x.split('_')[1])
            frame_paths.append(paths)

        df['frame_paths'] = frame_paths

        # obtain the maximum number of frames in all the videos
        self.max_frames = np.max([i for i in df['FramesPerVideo']])

        self.df = df
        df = shuffle(df, random_state = 5)

        train = df[:-25] # :152
        test = df[-25:] # 181:

        print("Train data: {}".format(len(train)))
        print("Test data: {}".format(len(test)))

        for column, data in train.iteritems():
            self.train[column] = data.values

        for column, data in test.iteritems():
            self.test[column] = data.values

        table = str.maketrans('', '', string.punctuation)
        table["\n"] = None

        # create a dictionary of all the words in the labels
        # the value for each word is going to be its index in the oneHot encoding 
        label_dict = {}
        i = 0
        for sentence in df['Label']:
            # print(sentence)
            for word in sentence.split():
                # convert the word into lowercase and remove all punctuations
                word = word.lower()
                word = word.translate(table) 

                # if the word doesn't exist in the dictionary already..
                # add it and increment the counter
                if word not in label_dict.keys():
                    label_dict[word] = i
                    i += 1

        self.labels_dictionary = label_dict
        self.num_classes = i + 1


        # self.print_dictionary()

    def get_trainData(self):
        return self.train 

    def get_testData(self):
        return self.test 

    def get_data(self):
        return self.train, self.test

    def get_num_frames(self, video_number):
        '''
        obtain the number of frames for a given video
        '''
        # get the row corresponding to this video
        data = self.df.loc[self.df['Video'] == video_number] 
        return np.squeeze(data['FramesPerVideo'].to_numpy())

    def get_frame_paths(self, video_number):
        '''
        returns a list of all the paths for each frame, in chronological order
        '''
        # get the row corresponding to this video
        data = self.df.loc[self.df['Video'] == video_number]
        paths = [i for p in data['frame_paths'] for i in p]
        return paths

    def print_label(self, video_number):
        # get the row corresponding to this video
        data = self.df.loc[self.df['Video'] == video_number] 

        print("The true label is: \"{}\"".\
                            format((data['Label']).to_string(index=False)))

    def get_label(self, video_number):
        '''
        obtain the true label for a given video
        '''
        # get the row corresponding to this video
        data = self.df.loc[self.df['Video'] == video_number] 
        # print(data['Label'])
        label = (data['Label']).to_string(index=False)
        print("Original - {}".format(label))
        
        table = str.maketrans('', '', string.punctuation)
        table["\n"] = None

        # remove punctuations
        label = label.translate(table)
        label = label.lower()
        print("Formatted - {}".format(label))

        return label

    def print_dictionary(self):
        '''
        print the contents of the dictionary
        '''
        for k, v in self.labels_dictionary.items():
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

    def get_oneHot(self, video_number):
        '''
        creates a vector from a given label such that if a word exists
        in the label, the vector at the word's index (from labels_dict)
        is 1 and others are all 0.

        for example, if the dictionary was:
            {'the' : 3,
             'dog' : 6,
             'ate' : 2}
        then the vector:
        [0 0 1 1 0 0 1 0 0 0] (assuming total number of words is 10)
        '''
        label = self.get_label(video_number)

        label_dict = self.labels_dictionary
        vec = np.zeros(self.num_classes)

        for word in label.split():
            vec[label_dict[word]] = 1

        return vec

    def get_matrix(self, num, model):
        '''
        returns a matrix of shape (2048, num_frames)
        where num_frames is the number of images extracted from the given video
        '''
        print("Getting the matrix for video {}".format(num))
        x = []
        paths = self.get_frame_paths(num)
        temp = np.zeros((2048, self.get_num_frames(num)))
        i = 0

        for p in paths:
            temp[:, i] = model.extract_features(p)
            i += 1

        return temp

    def padding(self, matrix):
        '''
        adds vectors of all 0's to all matrices whose columns is less than max_frames
        where max_frames is the max_frames for that type of data (training/test, etc.)
        '''
        max_frames = self.max_frames
        rows, cols = matrix.shape

        assert(cols <= max_frames)

        empty = np.zeros((2048, max_frames))
        empty[:rows, :cols] = matrix

        return empty

    def get_data_for_RNN(self):
        '''
        obtain the training and test data

        x_train is a np.array of shape (len(train), 2048, max_frames)
        x_test  is a np.array of shape (len(test),  2048, max_frames)
        '''
        # model = cnn.Inception_Model()

        ##################################################

        # for a given video in the dataset,
        # obtain a matrix of shape (2048, max_frames)
            # each column is a prediction from the pretrained CNN for each frame
            # column 0 is the (2048,)-vector for frame 0, etc.
        # stack up the matrices to obtain a gigantic matrix for the x_train
        # video_numbers = self.train['Video']

        # matrices = self.get_matrix(video_numbers[0], model)
        # matrices = self.padding(matrices)

        # for num in video_numbers[1:]:
        #     temp = self.get_matrix(num, model)
        #     temp = self.padding(temp)
        #     matrices = np.dstack((matrices, temp))

        # x_train = matrices.reshape(-1, 2048, self.max_frames)

        # print("Size of x training set: {}".format(x_train.shape))
        # np.save('x_train.npy', x_train)

        # ##################################################

        # # repeat above procedure for x_test
        # video_numbers = self.test['Video']
        # max_frames = np.max(self.test['FramesPerVideo'])

        # matrices = self.get_matrix(video_numbers[0], model)
        # matrices = self.padding(matrices)

        # for num in video_numbers[1:]:
        #     temp = self.get_matrix(num, model)
        #     temp = self.padding(temp)
        #     matrices = np.dstack((matrices, temp))

        # x_test = matrices.reshape(-1, 2048, self.max_frames)

        # print("Size of x testing set: {}".format(x_test.shape))
        # np.save('x_test.npy', x_test)

        ##################################################

        # for each video, obtain its true label
        # convert the label into its oneHot vector
        # stack up the vectors to get the matrix for y_train
        y_train = np.zeros((self.get_num_classes(), len(self.train['Video'])))
        video_numbers = self.train['Video']
        i = 0
        for num in video_numbers:
            y_train[:, i] = self.get_oneHot(num)

        np.save('y_train.npy', y_train)

        # repeat the above procedure for y_test
        y_test = np.zeros((self.get_num_classes(), len(self.test['Video'])))
        video_numbers = self.test['Video']
        i = 0
        for num in video_numbers:
            y_test[:, i] = self.get_oneHot(num)

        np.save('y_test.npy', y_test)

        print("Size of y training set: {}".format(y_train.shape))
        print("Size of y testing set: {}".format(y_test.shape))

        return x_train, y_train, x_test, y_test
