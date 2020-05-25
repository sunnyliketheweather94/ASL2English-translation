import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 
import os
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import optimizers, losses
from tensorflow.keras import models

class 3DCNN:
    def __init__(self):
        model = models.Sequential()
        model.add(Conv3D(32, kernel_size = (3, 3, 3), ))