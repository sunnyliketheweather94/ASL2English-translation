# used for processing the data
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob
import os
import cv2
from sklearn.utils import shuffle


# assume that we have labels already processed.
# labels are stored in 'labels.csv' where the first column is the video number
# and the second column is the corresponding label
class Dataset:
    '''
    Members: the following are for the train, dev and test data.
        num_frame: a list of the number of frames per video
                   the nth video's num_frame is num_frame[n - 1]
        file_names: a list of the file names of the videos
        frame_matrices: an array of len(file_names) containing the
                        3 normalized matrices of the pixel values
        labels: a list of the true labels

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
        self.dev = {}
        self.test = {}

        df = pd.read_csv(label_path)

        # add the frames_per_video to the dataframe
        frames_per_vid_path = os.path.join(data_path, 'framesPerVid.csv')
        frames_per_video = pd.read_csv(frames_per_vid_path)
        df['FramesPerVideo'] = frames_per_video['FramesPerVideo']

        # essentially rename the Translations column as the Label column
        df['Label'] = df['Translations']
        df.drop('Translations', axis = 1, inplace = True)

        # get the frame matrices for each video
        # path = '/Users/sunnyshah/Desktop/Spring 2020/CS 230/Project/Data/Processed Data'
        # path_list = [os.path.join(folder, s) for folder, subfolder, _ in os.walk(path) for s in sorted(subfolder)]
        # matrices_list = [] # to store the matrices for each video

        # for path in path_list:
        #     video_num = int(path.split('/')[-1]) # get the video number of current video
        #     matrices = np.empty([0])

        #     paths = []
        #     for filepath in glob.glob(path + '/*.jpg'):
        #         paths.append(filepath) # store the paths of all the frames in the subdirectory

        #     paths.sort(key = lambda x : x.split('_')[1])

        #     for p in paths:
        #         img = cv2.imread(p)
        #         img = img[:, 3:302, :]
        #         img = np.array(img) / 255
        #         print(img.shape)
        #         matrices = np.append(matrices, img)
        #     print(matrices.shape)

        #     matrices_list.append(matrices)

        #     print("Video {} has {} frames.".format(video_num, matrices.shape[0]))

        # df['frame_matrices'] = matrices_list



        self.df = df
        df = shuffle(df)

        print(df.head())

        train = df[:152]
        dev = df[152:178]
        test = df[178:]

        for column, data in train.iteritems():
            self.train[column] = data.values

        for column, data in dev.iteritems():
            self.dev[column] = data.values

        for column, data in test.iteritems():
            self.test[column] = data.values


    def get_trainData(self):
        return self.train 

    def get_devData(self):
        return self.dev 

    def get_testData(self):
        return self.test 

    def get_data(self):
        return self.train, self.dev, self.test

    def get_frames(self, video_number):
        # get the frame matrices for a given video
        data = self.df.loc[self.df['Video'] == video_number]

        return data['frame_matrices']

    def get_num_frames(self, video_number):
        # get the row corresponding to this video
        data = self.df.loc[self.df['Video'] == video_number] 

        return data['FramesPerVideo']

    def print_label(self, video_number):
        # get the row corresponding to this video
        data = self.df.loc[self.df['Video'] == video_number] 

        print("The true label is: \"{}\"".\
                            format((data['Label']).to_string(index=False)))


if __name__ == "__main__":
    labels_path = os.path.join('./Data', 'labels.csv')
    data_path = './Data/Processed Data/'

    data = Dataset(data_path, labels_path)
    train = data.get_testData()

    data.print_label(12)

