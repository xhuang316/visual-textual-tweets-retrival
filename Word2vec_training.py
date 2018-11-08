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



def CSV_to_Word2Vec_input(dir_of_CSV_file, column_number_of_text = 2, First_row_title = True):
    count  = 0
    word2vec_inputlist = []
    with open(dir_of_CSV_file) as f:
        readerf = csv.reader(f)
        if First_row_title:
            next(readerf)
        for row in readerf:
            temp_list = text_to_wordlist(row[2], remove_stopwords=True, stem_words=False, lemmatize_words =True)
            word2vec_inputlist.append(temp_list.split())
            count = count + 1
            if count % 10000 == 0:
                print ('loaded ' + str(count) + 'tweets')


    return (word2vec_inputlist)

#save trained model to a path

text_dir = 'D:\\_Work11\\Word2vec_vector_training\\results_Xiao_Huston_AllGeo.csv'

model_path = 'D:\\_Work11\\Word2Vec vector models\\' + 'test_model_v2'



input_to_Word2Vec = CSV_to_Word2Vec_input(text_dir, column_number_of_text=4, First_row_title=True)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Word2Vec(input_to_Word2Vec, window = 9, size = 300, min_count=5)
model.save(model_path)

model.wv.most_similar(positive=['woman', 'king'], topn=10)
