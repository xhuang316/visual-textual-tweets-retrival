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
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, KFold


positive_vector_path = 'D:\\_Work11\\Fused_model_training\\Fused_vector\\positive_fused_vector_trans.csv'
negative_vector_path = 'D:\\_Work11\\Fused_model_training\\Fused_vector\\negative_fused_vector_trans.csv'

# get x_train and y_train

x_train_list = []

with open(positive_vector_path, encoding="utf8") as f:
    readerf = csv.reader(f)
    for row in readerf:
        x_train_list.append(row)

with open(negative_vector_path, encoding="utf8") as f:
    readerf = csv.reader(f)
    for row in readerf:
        x_train_list.append(row)

x_train = np.array(x_train_list)
x_train = x_train.astype(float)

vector1 = [1] * 2092
vector2 = [0] * 2092
y_train_list = vector1 + vector2

y1_train = np.array(y_train_list)
y1_train = y1_train.astype(int)

print (x_train[1])
print (y1_train)


print (x_train.shape, y1_train.shape)





plt.figure(1)
#SVM

count = 1
kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)

for folded_train, folded_test in kfold.split(x_train, y1_train):

    X_train = x_train[folded_train]
    y_train = y1_train[folded_train]



    X_test = x_train[folded_test]
    y_test = y1_train[folded_test]

    # Logistic Regression
    LogR = LogisticRegression()
    LogR.fit(X_train, y_train)
    confidence = LogR.score(X_test, y_test)
    print ("fold:" + str(count) + "LogR: " + str(confidence))
    preds = LogR.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8,label= "fold:" + str(count) +"LogR"+'AUC = %0.3f' % (roc_auc))

    #Decision tree:
    DT = DecisionTreeClassifier(random_state=0)
    DT.fit(X_train, y_train)
    confidence = DT.score(X_test, y_test)
    print ("fold:" + str(count) + "DT: " + str(confidence))
    preds = DT.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8,label= "fold:" + str(count) +"DT"+'AUC = %0.3f' % (roc_auc))

    #Random forest:
    RF = RandomForestClassifier(random_state=0)
    RF.fit(X_train, y_train)
    confidence = RF.score(X_test, y_test)
    print ("fold:" + str(count) +"RF: " + str(confidence))
    preds = RF.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8,label= "fold:" + str(count) +"RF"+'AUC = %0.3f' % (roc_auc))

    #GaussianNB:
    GNB = GaussianNB()
    GNB.fit(X_train, y_train)
    confidence = GNB.score(X_test, y_test)
    print ("fold:" + str(count) +"GaussianNB: " + str(confidence))
    preds = GNB.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8,label="fold:" + str(count) + "GaussianNB"+'AUC = %0.3f' % (roc_auc))

    #MultinomialNB:
    MNB = MultinomialNB()
    MNB.fit(X_train, y_train)
    confidence = MNB.score(X_test, y_test)
    print ("fold:" + str(count) +"MultinomialNB: " + str(confidence))
    preds = MNB.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8,label= "fold:" + str(count) +"MultinomialNB"+'AUC = %0.3f' % (roc_auc))

    #BernoulliNB:
    BNB = BernoulliNB()
    BNB.fit(X_train, y_train)
    confidence = BNB.score(X_test, y_test)
    print ("fold:" + str(count) +"BernoulliNB: " + str(confidence))
    preds = MNB.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8,label= "fold:" + str(count) +"BernoulliNB"+'AUC = %0.3f' % (roc_auc))

    #K-means:
    n_clusters = len(np.unique(y_train))
    clf = KMeans(n_clusters = n_clusters, random_state=42)
    clf.fit(X_train)
    y_labels_train = clf.labels_
    y_labels_test = clf.predict(X_test)
    clf.score(X_test)
    print ("fold:" + str(count) +"K-means " + str(confidence))


    #LDA
    lda_classifier = LinearDiscriminantAnalysis()
    lda_classifier.fit(X_train, y_train)
    confidence = lda_classifier.score(X_test, y_test)
    print ("fold:" + str(count) +"LDA " + str(confidence))
    preds = lda_classifier.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8,label= "fold:" + str(count) +"LDA"+'AUC = %0.3f' % (roc_auc))

    #QuadraticDiscriminantAnalysis
    qda_classifier = QuadraticDiscriminantAnalysis()
    qda_classifier.fit(X_train, y_train)
    confidence = qda_classifier.score(X_test, y_test)
    print ("fold:" + str(count) +"QDA " + str(confidence))
    preds = qda_classifier.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8,label= "fold:" + str(count) +"QDA"+'AUC = %0.3f' % (roc_auc))


    #SVM
    for k in ['linear','poly','rbf','sigmoid']:
        clf = svm.SVC(kernel=k)
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print("fold:" + str(count) +'SVM_' + k,confidence)
        preds = clf.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, alpha=0.8,label= "fold:" + str(count) +"SVM"+ "(" + k + ")" + 'AUC = %0.3f' % (roc_auc))


    count = count + 1

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()