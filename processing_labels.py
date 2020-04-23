import os
import sys
import numpy as np
import pandas as pd
import string

def create_dictionary():
    file_name = "labels_volunteers.txt"
    table = str.maketrans('', '', string.punctuation)
    table["\n"] = None


    current_dir = os.getcwd()

    for folder in os.listdir(current_dir):
        if folder == "Data":
            path_ = os.path.join(current_dir, folder)

    os.chdir(path_)

    dict_ = {}
    labels = []
    i = 0

    # read each line at a time
    # for each line, split it into words
    # then remove punctuations from the words
    # and then create an entry in the dictionary 
    # if the word doesn't exist in the dictionary
    f = open(file_name, "r") # read file

    for line in f: 
        labels.append(line)

        for word in line.split():
            word = word.translate(table)

            if word not in dict_.keys():
                dict_[word] = i
                i += 1

    f.close()

    return dict_, labels

def create_vector(sentence, dict_):
    table = str.maketrans('', '', string.punctuation)
    vector = np.zeros(len(dict_))

    for word in sentence.split():
        word = word.translate(table)

        vector[dict_[word]] = 1

    return vector

if __name__ == "__main__":
    dictionary, labels = create_dictionary()

    vec_ = create_vector(labels[24], dictionary)
    print(vec_)