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
from sklearn.model_selection import StratifiedKFold, KFold

tweets_training_path = 'D:\\_Work11\\Word2Vec_training\\Word2Vec_training_tweets_with_labels_with_noise.csv'
#tweets_training_path = 'D:\\_Work11\\Word2Vec_training\\Word2Vec_training_tweets_with_labels.csv'
word2vec_model_path = 'D:\\_Work11\\Word2Vec vector models\\' + 'test_model_v2'



text_column = 2
label_column = 8


x_train = CSV_to_x_train(tweets_training_path,word2vec_model_path, text_column)
y_train = CSV_to_y_train_not_one_hot_vector(tweets_training_path, label_column)
maxlen = CSV_get_maxlen(tweets_training_path, text_column)
dim = get_word2vec_dim(word2vec_model_path)

print (x_train.shape)
print (y_train.shape)
# model parameters

seed = 124
np.random.seed(seed)
nb_K_Fold = 5
kfold = StratifiedKFold(n_splits = nb_K_Fold, shuffle=True, random_state=seed)

cvscores = []
count = 1

acc_list = []
loss_list = []
val_acc_list = []
val_loss_list = []

nb_feature_maps = 64
BATCH_SIZE = 7500
ngram_filters = [5, 4, 3, 2]
epochs = 100

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for folded_train, folded_test in kfold.split(x_train, y_train):



    model_input = Input(shape=(maxlen, dim, 1))
    flattened_tensors = []


    model_input = Input(shape= (maxlen, dim, 1))
    for n_gram in ngram_filters:
        layer1 = Convolution2D(nb_feature_maps, (n_gram, dim), padding= 'valid', input_shape=(maxlen, dim, 1),activation='relu')(model_input)
        layer2 = MaxPooling2D(pool_size=(maxlen - n_gram + 1, 1))(layer1)
        layer3 = Flatten()(layer2)
        flattened_tensors.append(layer3)


    conc_tensor = concatenate(flattened_tensors, axis=-1)
    D_conc_tensor = Dropout(0.5)(conc_tensor)
    out = Dense(2,activation='softmax')(D_conc_tensor)
    W2V_model = Model(inputs= [model_input],outputs = [out])

    #model.summary()
    sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0.9,nesterov=True)
    W2V_model.compile(loss='categorical_crossentropy',optimizer = sgd,  metrics=['accuracy'])

    callbacks = callback_generator(Earlystop=False, Reduce_lr=True)




    print ('fitting model')

    CNN_model = W2V_model.fit(x_train[folded_train], np_utils.to_categorical(y_train[folded_train], 2), epochs=epochs, batch_size=BATCH_SIZE, shuffle=True, validation_split= 0.2, callbacks=callbacks)

    W2V_model.save('D:\\_Work11\\Text_training_model_K_fold\\' + 'flood' + str(count) + 'CNN')

    acc = CNN_model.history['acc']
    val_acc = CNN_model.history['val_acc']
    loss = CNN_model.history['loss']
    val_loss = CNN_model.history['val_loss']
    acc_list.append(acc)
    val_acc_list.append(val_acc)
    loss_list.append(loss)
    val_loss_list.append(val_loss)

    scores = W2V_model.evaluate(x_train[folded_test], np_utils.to_categorical(y_train[folded_test], 2), verbose=1)
    print("%s: %.2f%%" % (W2V_model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)



    y_score = W2V_model.predict(x_train[folded_test])

# convert y_score to single probability corresponding to each label

    y_score_positive = y_score[:,1]


    fpr, tpr, thresholds = roc_curve(y_train[folded_test], y_score_positive, pos_label= 1)


    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.figure(1)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8,
             label='ROC fold %d (AUC = %0.3f)' % (count, roc_auc))

    count = count + 1

# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)

# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + 3*std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - 3*std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.5,
#                  label=r'$\pm$ 3 std. dev.')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")



epochs_list = range(1, epochs+1)


plt.figure(2)
for i in range(nb_K_Fold-1):
    plt.plot(epochs_list, acc_list[i], 'blue', label='Training acc')
    plt.plot(epochs_list, val_acc_list[i], 'red', label='Validation acc')

plt.figure(3)
for i in range(nb_K_Fold-1):
    plt.plot(epochs_list, loss_list[i], 'blue', label='Training loss')
    plt.plot(epochs_list, val_loss_list[i], 'red', label='Validation loss')
plt.show()

for score in cvscores:
    print ("acc: " + str(score))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))




