import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
import cv2
from sklearn.model_selection import StratifiedKFold, KFold
from Functions import add_new_last_layer_InceptionV3, setup_to_finetune_InceptionV3, setup_to_transfer_learn_InceptionV3
from sklearn.model_selection import train_test_split
import gc


IM_WIDTH, IM_HEIGHT = 299, 299
nb_channels = 3
nb_K_Fold = 2
seed = 124

#Reading the picture from two different folders
Flooding_picture_dir = 'D:\\_Work11\\Picture_training_dataset\\Flooding'
Non_flooding_picture_dir = 'D:\\_Work11\\Picture_training_dataset\\Non_flooding'
model_performance_save_path = 'D:\\_Work11\\Picture_training_model_performance'



imList = []
labelList = []
count = 0
Flooding_picture_list = os.listdir(Flooding_picture_dir)
for Flooding_picture in Flooding_picture_list:
    abs_path = Flooding_picture_dir + '\\' + Flooding_picture
    label = 1
    labelList.append(label)
    img = cv2.imread(abs_path)
    img = cv2.resize(img,(IM_WIDTH, IM_HEIGHT), 0,0, cv2.INTER_LINEAR)
    imList.append(img)
    count  = count + 1

    if count % 1000 ==0:
        print ('loaded ' + str(count) + ' images')

Non_Flooding_picture_list = os.listdir(Non_flooding_picture_dir)
for Non_Flooding_picture in Non_Flooding_picture_list:
    abs_path = Non_flooding_picture_dir + '\\' + Non_Flooding_picture
    label = 0
    labelList.append(label)
    img = cv2.imread(abs_path)
    img = cv2.resize(img,(IM_WIDTH, IM_HEIGHT), 0,0, cv2.INTER_LINEAR)
    imList.append(img)
    count = count + 1
    if count % 1000 ==0:
        print ('loaded ' + str(count) + ' images')

x_train = np.array(imList)
x_train = np.array(x_train, dtype='float')/255.0
x_train = x_train.reshape(x_train.shape[0], IM_WIDTH, IM_WIDTH, nb_channels)
y_train = np.array(labelList)


x_train, X_test, y_train, Y_test = train_test_split(x_train, y_train, test_size=0, random_state=42, shuffle= True)


print (y_train.shape)
print (x_train.shape)

# preparing K-folds and plot parameters

cvscores = []
count = 1

total_acc_list = []
total_loss_list = []
total_val_acc_list = []
total_val_loss_list = []





BATCH_SIZE = 64
epochs = 2
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
callbacks = callback_generator(Earlystop=False, Reduce_lr=False)

kfold = StratifiedKFold(n_splits = nb_K_Fold, shuffle=True, random_state=seed)

for folded_train, folded_test in kfold.split(x_train, y_train):

    # compile and fit model right here
    fold_performance = []



    base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    model = add_new_last_layer_InceptionV3(base_model, 2)

    setup_to_transfer_learn_InceptionV3(model, base_model)
    transfer_model = model.fit(x_train[folded_train], np_utils.to_categorical(y_train[folded_train], 2), epochs=epochs, batch_size=BATCH_SIZE, shuffle=True, validation_split= 0.2, callbacks=callbacks)


    transfer_acc = transfer_model.history['acc']
    transfer_val_acc = transfer_model.history['val_acc']
    transfer_loss = transfer_model.history['loss']
    transfer_val_loss = transfer_model.history['val_loss']

    setup_to_finetune_InceptionV3(model)
    tune_model = model.fit(x_train[folded_train], np_utils.to_categorical(y_train[folded_train], 2), epochs=epochs, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.2, callbacks=callbacks)


    tune_acc = tune_model.history['acc']
    tune_val_acc = tune_model.history['val_acc']
    tune_loss = tune_model.history['loss']
    tune_val_loss = tune_model.history['val_loss']


    total_acc = transfer_acc + tune_acc
    total_val_acc = transfer_val_acc + tune_val_acc
    total_loss = transfer_loss + tune_loss
    total_val_loss = transfer_val_loss + tune_val_loss

    temp = []

    temp.append(total_acc)
    temp.append(total_val_acc)
    temp.append(total_loss)
    temp.append(total_val_loss)

    temp_trans = list(map(list, zip(*temp)))

    abs_path = model_performance_save_path + '\\fold'+ str(count) + '.csv'

    with open(abs_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(temp_trans)




    total_acc_list.append(total_acc)
    total_val_acc_list.append(total_val_acc)
    total_loss_list.append(total_loss)
    total_val_loss_list.append(total_val_loss)




    model_path = ('D:\\_Work11\\Picture_training_model_N_fold\\' + str(count) + '-fold')
    model.save(model_path)

    scores = model.evaluate(x_train[folded_test], np_utils.to_categorical(y_train[folded_test], 2), verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    y_score = model.predict(x_train[folded_test])

    # convert y_score to single probability corresponding to each label

    y_score_positive = y_score[:, 1]

    fpr, tpr, thresholds = roc_curve(y_train[folded_test], y_score_positive, pos_label=1)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.figure(1)
    plt.plot(fpr, tpr, lw=1, alpha=0.5,
             label='ROC fold %d (AUC = %0.2f)' % (count, roc_auc))

    print('----------------------in ' + str(count) + '-folds------------------')
    count = count + 1

    del model
    del transfer_model
    del tune_model
    gc.collect()



mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + 3*std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - 3*std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.5,
                 label=r'$\pm$ 3 std. dev.')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")



epochs_list = range(1, (epochs)*2 + 1)




plt.figure(2)
for i in range(nb_K_Fold):
    plt.plot(epochs_list, total_acc_list[i], 'blue', label='Training acc')
    plt.plot(epochs_list, total_val_acc_list[i], 'red', label='Validation acc')

plt.figure(3)
for i in range(nb_K_Fold):
    plt.plot(epochs_list, total_loss_list[i], 'blue', label='Training loss')
    plt.plot(epochs_list, total_val_loss_list[i], 'red', label='Validation loss')
plt.show()

for score in cvscores:
    print ("acc: " + str(score))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))






