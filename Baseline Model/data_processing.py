# used for processing the data
import pandas as pd 
import glob
import os
import cv2
from sklearn.utils import shuffle

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
        self.dev = {}
        self.test = {}
        self.img_shape = (231, 299, 3)

        df = pd.read_csv(label_path)

        # add the frames_per_video to the dataframe
        frames_per_vid_path = os.path.join(data_path, 'framesPerVid.csv')
        frames_per_video = pd.read_csv(frames_per_vid_path)
        df['FramesPerVideo'] = frames_per_video['FramesPerVideo']

        # essentially rename the Translations column as the Label column
        df['Label'] = df['Translations']
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

        self.df = df
        df = shuffle(df)

        train = df[:152]
        dev = df[152:178]
        test = df[178:]

        print("Train data: {}".format(len(train)))
        print("Dev data: {}".format(len(dev)))
        print("Test data: {}".format(len(test)))

        for column, data in train.iteritems():
            self.train[column] = data.values

        for column, data in dev.iteritems():
            self.dev[column] = data.values

        for column, data in test.iteritems():
            self.test[column] = data.values

        table = str.maketrans('', '', string.punctuation)
        table["\n"] = None

        label_dict = {}
        i = 0
        for sentence in df['labels']:
            for word in sentence.split():
                word = word.lower()

                word = word.translate(table) # remove all punctuations

                if word not in label_dict.keys():
                    label_dict[word] = i
                    i += 1

        self.labels_dictionary = label_dict


    def get_trainData(self):
        return self.train 

    def get_devData(self):
        return self.dev 

    def get_testData(self):
        return self.test 

    def get_data(self):
        return self.train, self.dev, self.test

    def get_num_frames(self, video_number):
        # get the row corresponding to this video
        data = self.df.loc[self.df['Video'] == video_number] 
        return data['FramesPerVideo'].to_numpy()

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
        # get the row corresponding to this video
        data = self.df.loc[self.df['Video'] == video_number] 
        return (data['Label']).to_string(index=False)

    def get_num_classes(self):
        return len(self.labels_dictionary)

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
        label = get_label(video_number)

        label_dict = self.labels_dictionary
        vec = np.zeros(len(label_dict))

        for word in label.split():
            vec[label_dict[word]] = 1

        return vec

    def get_data_for_RNN(self):
        model = cnn.Inception_Model()

        x_train = []
        video_numbers = self.train['Video']
        for num in video_numbers:
            paths = get_frame_paths(num) # get list of paths to all images for video
            temp = np.zeros(get_num_frames(num), 2048)
            i = 0

            for p in paths:
                temp[i, :] = model.extract_features(p)
                i += 1

            x_train.append(temp)

        x_train = np.array(x_train)

        x_test = []
        video_numbers = self.test['Video']
        for num in video_numbers:
            paths = get_frame_paths(num) # get list of paths to all images for video
            temp = np.zeros(get_num_frames(num), 2048)
            i = 0

            for p in paths:
                temp[i, :] = model.extract_features(p)
                i += 1

            x_test.append(temp)

        x_test = np.array(x_test)

        x_dev = []
        video_numbers = self.dev['Video']
        for num in video_numbers:
            paths = get_frame_paths(num) # get list of paths to all images for video
            temp = np.zeros(get_num_frames(num), 2048)
            i = 0

            for p in paths:
                temp[:, i] = model.extract_features(p)
                i += 1

            x_dev.append(temp)

        x_dev = np.array(x_dev)


        y_train = np.zeros((get_num_classes(), len(self.train['Video'])))
        video_numbers = self.train['Video']
        i = 0
        for num in video_numbers:
            y_train[:, i] = get_oneHot(num)

        y_test = np.zeros((get_num_classes(), len(self.test['Video'])))
        video_numbers = self.test['Video']
        i = 0
        for num in video_numbers:
            y_test[:, i] = get_oneHot(num)

        y_dev = np.zeros((get_num_classes(), len(self.dev['Video'])))
        video_numbers = self.train['Video']
        i = 0
        for num in video_numbers:
            y_dev[:, i] = get_oneHot(num)

        return x_train, y_train, x_dev, y_dev, x_test, y_test
