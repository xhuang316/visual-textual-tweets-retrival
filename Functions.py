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
from keras.preprocessing.sequence import pad_sequences
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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam

def text_to_wordlist(text, remove_stopwords=False, stem_words=False, lemmatize_words = False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)


    # Clean the text
    text = re.sub(r"https\S+ ", "", text)
    text = re.sub(r"https\S+", "", text)
    text = re.sub(r"@\S+ ", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+ ", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", "", text)
    text = re.sub(r"\/", "", text)
    text = re.sub(r"\^", "", text)
    text = re.sub(r"\+", "", text)
    text = re.sub(r"\-", "", text)
    text = re.sub(r"\=", "", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"\-", "", text)
    text = re.sub(r"\;", "", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    if lemmatize_words:
        text = text.split()
        lemmer = WordNetLemmatizer()
        lemmer_words = [lemmer.lemmatize(word) for word in text]
        text = " ".join(lemmer_words)


    # Return a list of words
    return (text)

def CSV_to_x_train(csvfile, word2vec_model_path, column_number_text):
    word2vec_model = Word2Vec.load(word2vec_model_path)

    text_list = []
    label_list = []
    # get the dimension for Word2vec model
    dim = word2vec_model.wv.vector_size

    # convert to initial input vector
    conc_sentence_vec = []
    with open(csvfile) as f:
        readerf = csv.reader(f)
        for row in readerf:
            sentence_vec = []
            tokenized_text = text_to_wordlist(row[column_number_text], remove_stopwords=True,
                                                        stem_words=False, lemmatize_words=True)
            tokenized_text_list = tokenized_text.split()
            text_list.append(tokenized_text_list)
            for word in tokenized_text_list:
                if word in word2vec_model.wv.vocab:
                    # append word vector to sentence
                    sentence_vec.append(word2vec_model[word])
                else:
                    # randomize a vector
                    sentence_vec.append(np.random.uniform(-0.25, 0.25, dim))
            conc_sentence_vec.append(sentence_vec)

    # get the max length of sentence

    length = []
    for x in text_list:
        length.append(len(x))
    maxlen = max(length)
    print (maxlen)

    # 0 padding with initial vector and give it to x_train

    conc_sentence_vec_padded = pad_sequences(conc_sentence_vec, maxlen=maxlen, dtype='float32', padding='post',
                                             truncating='pre', value=0.0)
    x_train = conc_sentence_vec_padded
    x_train = np.array(x_train)
    # reshape x and add another fake channel dimension
    x_train = x_train.reshape(x_train.shape[0], maxlen, dim, 1)

    return (x_train)

def CSV_to_y_train(csvfile, column_number_label):
    label_list = []
    with open(csvfile) as f:
        readerf = csv.reader(f)
        for row in readerf:
            label_list.append(int(row[column_number_label]))


    y_train = np.array(label_list)
    y_train = np_utils.to_categorical(y_train, 2)

    print (y_train.shape)

    return (y_train)

def CSV_to_y_train_not_one_hot_vector(csvfile, column_number_label):
    label_list = []
    with open(csvfile) as f:
        readerf = csv.reader(f)
        for row in readerf:
            label_list.append(int(row[column_number_label]))


    y_train = np.array(label_list)

    return (y_train)





def CSV_get_maxlen(csvfile, column_number_text):
    text_list = []
    with open(csvfile) as f:
        readerf = csv.reader(f)
        for row in readerf:
            tokenized_text = text_to_wordlist(row[column_number_text], remove_stopwords=True,
                                              stem_words=False, lemmatize_words=True)
            tokenized_text_list = tokenized_text.split()
            text_list.append(tokenized_text_list)

    # get the max length of sentence
    length = []
    for x in text_list:
        length.append(len(x))
    maxlen = max(length)

    return (maxlen)



def get_word2vec_dim(word2vec_path):
    word2vec_model = Word2Vec.load(word2vec_path)
    dim = word2vec_model.wv.vector_size
    return (dim)


def callback_generator(Earlystop = False, Reduce_lr = False):
    callback_list = []
    if Earlystop:
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5,
                              verbose=1, mode='auto')
        callback_list.append(earlystop)

    if Reduce_lr:
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.3,
                                  patience=5, min_lr=0.000001)
        callback_list.append(reduce_lr)

    return (callback_list)

def add_new_last_layer_InceptionV3(base_model, nb_classes):

  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x) #new FC layer, random init
  x = Dropout(0.2)(x)
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model

sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-7,  nesterov=True)
RMS = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=False)
def setup_to_transfer_learn_InceptionV3(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
      layer.trainable = False
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


def setup_to_finetune_InceptionV3(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:172]:
      layer.trainable = False
  for layer in model.layers[172:]:
      layer.trainable = True
  model.compile(optimizer=SGD(lr=0.00005, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def gettweetID_Text(csvfile):
    tweetID_Text_list = []
    with open(csvfile, encoding="utf8") as f:
        next(f)
        readerf = csv.reader(f)
        for row in readerf:
            if row[7] == '1-Text':
                tweetID_Text_list.append(int(row[0]))

    return tweetID_Text_list

def gettweetID_Photo(csvfile):
    tweetID_Text_list = []
    with open(csvfile, encoding="utf8") as f:
        next(f)
        readerf = csv.reader(f)
        for row in readerf:
            if row[7] == '2-Photo':
                tweetID_Text_list.append(int(row[0]))

    return tweetID_Text_list

def gettweetID_Both(csvfile):
    tweetID_Text_list = []
    with open(csvfile, encoding="utf8") as f:
        next(f)
        readerf = csv.reader(f)
        for row in readerf:
            if row[7] == '3-Both':
                tweetID_Text_list.append(float(row[0]))

    return tweetID_Text_list


def gettweetID_All_Positive(csvfile):
    tweetID_Text_list = []
    with open(csvfile, encoding="utf8") as f:
        next(f)
        readerf = csv.reader(f)
        for row in readerf:
            if row[7] == '3-Both' or row[7] == '2-Photo' or row[7] == '1-Text':
                tweetID_Text_list.append(row[0])

    return tweetID_Text_list

def CSV_to_x_test(csvfile, word2vec_model_path, column_number_text):
    word2vec_model = Word2Vec.load(word2vec_model_path)

    text_list = []
    label_list = []
    # get the dimension for Word2vec model
    dim = word2vec_model.wv.vector_size

    # convert to initial input vector
    conc_sentence_vec = []
    with open(csvfile) as f:
        readerf = csv.reader(f)
        for row in readerf:
            sentence_vec = []
            tokenized_text = text_to_wordlist(row[column_number_text], remove_stopwords=True,
                                                        stem_words=False, lemmatize_words=True)
            tokenized_text_list = tokenized_text.split()
            text_list.append(tokenized_text_list)
            for word in tokenized_text_list:
                if word in word2vec_model.wv.vocab:
                    # append word vector to sentence
                    sentence_vec.append(word2vec_model[word])
                else:
                    # randomize a vector
                    sentence_vec.append(np.random.uniform(-0.25, 0.25, dim))
            conc_sentence_vec.append(sentence_vec)

    maxlen = 25


    # 0 padding with initial vector and give it to x_train

    conc_sentence_vec_padded = pad_sequences(conc_sentence_vec, maxlen=maxlen, dtype='float32', padding='post',
                                             truncating='pre', value=0.0)
    x_train = conc_sentence_vec_padded
    x_train = np.array(x_train)
    # reshape x and add another fake channel dimension
    x_train = x_train.reshape(x_train.shape[0], maxlen, dim, 1)

    return (x_train)




