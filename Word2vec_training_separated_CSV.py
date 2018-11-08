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
import logging
from Functions import text_to_wordlist
import os

csv.field_size_limit(1000000000)

Separated_CSV_folder = 'D:\\_Work11\\Word2vec_vector_training\\Training_with_separated_CSV'
model_path = 'D:\\_Work11\\Word2Vec vector models\\' + 'test_model_v2'
CSV_list = os.listdir(Separated_CSV_folder)


count = 0
word2vec_inputlist = []

for CSVfile in CSV_list:
    abs_path = Separated_CSV_folder + '\\' + CSVfile

    with open(abs_path) as f:
        readerf = csv.reader(f)
        next(readerf)
        for row in readerf:
            try:

                row_value = row[4]

            except IndexError:
                 row_value = 'none'

            temp_list = text_to_wordlist(row_value, remove_stopwords=True, stem_words=False, lemmatize_words =True)
            word2vec_inputlist.append(temp_list.split())
            count = count + 1
            if count % 10000 == 0:
                print ('loaded ' + str(count) + 'tweets')




#save trained model to a path



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Word2Vec(word2vec_inputlist, window = 9, size = 300, min_count=5)
model.save(model_path)
