import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import os
import sys
import requests
import cv2
import string


data_dir = os.path.join('/Users/sunnyshah/Desktop/Spring 2020/CS 230/Project/3D_CNN', 'data')
original_dir = '/Users/sunnyshah/Desktop/Spring 2020/CS 230/Project/3D_CNN'

os.chdir(data_dir)

def FrameCapture(path): 
    vidObj = cv2.VideoCapture(path) # Path to video file 
    count = 1 # Used as counter variable  
    success = 1 # checks whether frames were extracted
  
    while success:                      # vidObj object calls read 
        success, image = vidObj.read()  # function extract frames 
        if success:
            image = image[:320, 82:232] # image is of shape (320, 150, 3)
            cv2.imwrite("frame_%d.jpg" % count, image) # Saves the frames with frame-counts
            count += 1

os.chdir('/Users/sunnyshah/Desktop/Spring 2020/CS 230/Project/3D_CNN')
data = './data/comprehensive.xlsx'
df = pd.read_excel(data, header = None) # get the excel file
df = df.iloc[np.random.permutation(len(df))] # shuffle it randomly


data = df[:351] # extract the first 900 videos; 
# some will be discarded if they're compound signs (aka 2+ signs put together to make a phrase)

print(data)


labels = data[2]
links = data[17]

set_labels = set([string for string in labels])
# labels = [w for w in labels]

# # remove punctuations 
# # DOESN'T WORK
# table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
# labels = [w.translate(table) for w in labels]

# trial = [w for w in trial]

# print(trial)
print(set_labels)


# links for all videos for a given label, stored in a dictionary
links_dict = {label: data.loc[data[2] == label, 17] for label in labels}

print(links_dict)


true_labels = {}
counter = 1

for label, urls in links_dict.items():    
    for url in urls:
        video_number = 'video ' + str(counter)
        
        os.chdir(data_dir)
        
        word = ''.join([c.lower() for c in label if c.isupper()]) # removes all lower-case letters, becomes our label
        print("Video number - {}\tLabel - {}\tOriginal label - {}".format(counter, word, label))

        os.mkdir(video_number)
        os.chdir(video_number) # make a folder for that label and change the directory to that folder

        true_labels[str(counter)] = [word]

        # download the video and save in the folder for a given label
        r = requests.get(url, allow_redirects = True)
        vid_name = 'video' + str(counter) + '.mov'
        open(vid_name, 'wb').write(r.content)

        vid_path = os.path.join(os.getcwd(), vid_name)
        FrameCapture(vid_path)

        counter += 1
        

true_labels = pd.DataFrame(true_labels)
os.chdir(data_dir)


true_labels_transposed = true_labels.transpose()
true_labels_transposed.reset_index(level = 0, inplace=True)
true_labels_transposed.columns = ['Video number', 'Label']
print(true_labels_transposed)

true_labels_transposed.to_csv('labels.csv', index = False)


