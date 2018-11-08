import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras import optimizers
import matplotlib.pylab as plt
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Dropout, Flatten, Conv2D, Merge, concatenate, Input
from keras.models import Sequential
from keras.models import Model
from Functions import CSV_to_x_train, CSV_to_y_train, CSV_get_maxlen,get_word2vec_dim, CSV_to_y_train_not_one_hot_vector, callback_generator
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Model, load_model
from scipy import interp
import time
from sklearn.datasets import make_classification
from sklearn.cross_validation import KFold
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from gensim.models.word2vec import Word2Vec
import csv
from gensim.models import Word2Vec
import nltk
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import Functions
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
import cv2
from sklearn.model_selection import StratifiedKFold, KFold
from Functions import add_new_last_layer_InceptionV3, setup_to_finetune_InceptionV3, setup_to_transfer_learn_InceptionV3, gettweetID_Text, gettweetID_Photo, gettweetID_Both, text_to_wordlist,gettweetID_All_Positive,CSV_to_x_test
from sklearn.model_selection import train_test_split
import gc



word2vec_model_path = 'D:\\_Work11\\Word2Vec vector models\\test_model_v2'
Textual_model_path = 'D:\\_Work11\\Text_training_model_1_fold\\Flood_CNN'
Visual_model_path = 'D:\\_Work11\\Picture_training_model_1_fold\\1_fold'
Positive_picture_path = 'D:\\_Work11\\Fused_model_training\\positive'

# model load and extract 1024 layer:

V_model = load_model(Visual_model_path)
T_model= load_model(Textual_model_path)

V_model.layers.pop()
V_model.outputs = [V_model.layers[-1].output]
V_model.layers[-1].outbound_nodes = []


T_model.layers.pop()
T_model.outputs = [T_model.layers[-1].output]
T_model.layers[-1].outbound_nodes = []


Positive_tweet_path = 'D:\\_Work11\\Fused_model_training\\Positive_TweetID.csv'


# generate N 1024 textual vector
P_textual_tensor = CSV_to_x_test(Positive_tweet_path, word2vec_model_path, 2)
P_textual_vector = T_model.predict(P_textual_tensor)
P_textual_vector = P_textual_vector.tolist()

# generate N 1024 visual vector

IM_WIDTH = 299
IM_HEIGHT = 299
count1 = 0
count2 = 0
P_visual_vector = []

     # built picture ID list
PictureID_list = []
Positive_picture_list = os.listdir(Positive_picture_path)
for Positive_picture in Positive_picture_list:
    PictureID = Positive_picture.split('.')[0]
    PictureID_list.append(PictureID)

with open(Positive_tweet_path, encoding="utf8") as f:
    readerf = csv.reader(f)
    for row in readerf:
        if row[0] in PictureID_list:
            print (row[0], row[2])
            imList = []
            abs_path = Positive_picture_path + '\\' + row[0] + '.jpg'
            img = cv2.imread(abs_path)
            img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT), 0, 0, cv2.INTER_LINEAR)
            imList.append(img)
            P_picture_tensor = np.array(imList)
            P_picture_tensor = np.array(P_picture_tensor, dtype='float') / 255.0
            P_picture_tensor = P_picture_tensor.reshape(P_picture_tensor.shape[0], 299, 299, 3)
            vector = V_model.predict(P_picture_tensor)
            vector = vector[0].tolist()
            P_visual_vector.append(vector)
            count1 = count1 + 1
            print (count1)
        else:
            vector = [0] * 1024
            count2 = count2 + 1
            P_visual_vector.append(vector)


# concatenate two vectors to a long vector and save to CSV
P_outpath =  'D:\\_Work11\\Fused_model_training\\Fused_vector\\positive_fused_vector_trans.csv'
P_fused_vector = []
for i in range(len(P_textual_vector)):
    vector = P_textual_vector[i] + P_visual_vector[i]
    P_fused_vector.append(vector)


#P_fused_vector_trans = list(map(list, zip(*P_fused_vector)))
with open(P_outpath, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(P_fused_vector)












