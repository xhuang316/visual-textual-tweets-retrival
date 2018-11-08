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
from Functions import CSV_to_x_train, CSV_to_y_train, CSV_get_maxlen,get_word2vec_dim, CSV_to_y_train_not_one_hot_vector

tweets_training_path = 'D:\\_Work11\\Word2Vec_training\\Word2Vec_training_tweets_with_labels.csv'
word2vec_model_path = 'D:\\_Work11\\Word2Vec vector models\\' + 'test_model_v2'


text_column = 2
label_column = 8


x_train = CSV_to_x_train(tweets_training_path,word2vec_model_path, text_column)
y_train = CSV_to_y_train(tweets_training_path, label_column)
maxlen = CSV_get_maxlen(tweets_training_path, text_column)
dim = get_word2vec_dim(word2vec_model_path)

print (x_train.shape)
print (y_train.shape)
# model parameters
nb_feature_maps = 64
BATCH_SIZE = 7500
epochs = 200
ngram_filters = [5,4,3,2]

model_input = Input(shape=(maxlen, dim, 1))
flattened_tensors = []

for n_gram in ngram_filters:
    layer1 = Convolution2D(nb_feature_maps, (n_gram, dim), padding= 'valid', input_shape=(maxlen, dim, 1),activation='relu')(model_input)
    layer2 = MaxPooling2D(pool_size=(maxlen - n_gram + 1, 1))(layer1)
    layer3 = Flatten()(layer2)
    flattened_tensors.append(layer3)

#conc_tensor = concatenate(flattened_tensors, axis=-1)
#D_conc_tensor = Dropout(0.5)(conc_tensor)
#second_last_layer = Dense(nb_feature_maps * len(ngram_filters))(D_conc_tensor)
#out = Dense(2)(second_last_layer)


conc_tensor = concatenate(flattened_tensors, axis=-1)
D_conc_tensor = Dropout(0.5)(conc_tensor)
out = Dense(2,activation='softmax')(D_conc_tensor)

model = Model(inputs= [model_input],outputs = [out])

model.summary()
sgd = optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer = sgd,  metrics=['accuracy'])
print ('fitting model')
Model = model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, validation_split=0.3, shuffle= True)
#model.save('D:\\_Work11\\Text_training_model_1_fold\\Flood_CNN')

acc = Model.history['acc']
val_acc = Model.history['val_acc']
loss = Model.history['loss']
val_loss = Model.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'blue', label= 'Training acc')
plt.plot(epochs,val_acc,'red', label= 'Validation acc')
#plt.title('Training and validation accuracy')
#plt.legend()
plt.figure()
plt.plot(epochs, loss, 'blue',label='Training loss')
plt.plot(epochs, val_loss, 'red',label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
plt.show()

