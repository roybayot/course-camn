
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import sys
import os

import sklearn
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics

reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
sys.setdefaultencoding("UTF-8")


# In[2]:

def clean_text(raw_text):
    review_text = BeautifulSoup(raw_text).get_text()
    words = review_text.lower().split()
    return(" ".join(words))


# In[3]:

languages = ["english"]
datafiles = ["summary-english-truth.txt"]


# In[4]:

for language, datafile in zip(languages, datafiles):
    train = pd.read_csv(datafile, header=0, delimiter="\t", quoting=1)
    num_text = train["text"].size
    clean_train_data = []
    
    for i in xrange( 0, num_text):
        clean_train_data.append( clean_text( train["text"][i] ) )
    
    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None)
    train_x = vectorizer.fit_transform(clean_train_data)
    train_x = train_x.toarray()


# In[10]:

train = pd.read_csv("new-train-data.tsv", header=0, delimiter="\t", quoting=1)
train_y = train['target']


# In[11]:

with open('combo-important-features.txt') as f:
    alist = [line.rstrip() for line in f]
all_indices_ranked = alist[0].split(',')
all_indices_ranked = [int(x) for x in all_indices_ranked]
all_indices_ranked = [x-1 for x in all_indices_ranked]


# In[12]:

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


# In[13]:

X_train, X_test, y_train, y_test = train_test_split( train_x, train_y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


# In[14]:

scores = ['accuracy']


# In[16]:

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10,
                       scoring= score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# In[ ]:



