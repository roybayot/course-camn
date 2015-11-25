#!/usr/bin/python

"""
Experiment 01:
    4 languages
    use tfidf
        put everything in lowercase
        did not remove stopwords
        no other preprocessing
"""

print(__doc__)

import os
import sys
import pandas as pd
import numpy as np
import re
import timeit

import datetime
import xlsxwriter

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


# settings:
# n_words = 10000
n_folds = 10

def review_to_words(raw_review):
    # function to convert a raw review to a string of words
    # the input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    # 1. remove html
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. remove non-letters        
    # letters_only = re.sub("[^a-za-z]", " ", review_text) 
    #
    # 3. convert to lower case, split into individual words
    # words = letters_only.lower().split()                             
    words = review_text.lower().split()                             
    #
    # 4. in python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # stops = set(stopwords.words("english"))                  
    # 
    # 5. remove stop words
    # meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. join the words back into one string separated by space, 
    # and return the result.
    # return( " ".join( meaningful_words ))
    return( " ".join( words ))



def main(argv):
    languages = ["english", "dutch", "italian", "spanish"]
    datafiles = ["summary-english-truth.txt", \
		 "summary-dutch-truth.txt", \
		 "summary-italian-truth.txt", \
		 "summary-spanish-truth.txt"]
    tasks = ["gender",
             "age",
             "extroverted",
             "stable",
             "agreeable",
             "open",
             "conscientious"]

    n_folds = 10

    filename = str(datetime.datetime.now()) + '-experiment-01.xlsx'
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    worksheet.set_column('A:C', 20)

    index = 0
    for language, datafile in zip(languages, datafiles):
        train = pd.read_csv(datafile, header=0, delimiter="\t", quoting=1)
        num_text = train["text"].size
        
        clean_train_data = []

        for i in xrange( 0, num_text):
            clean_train_data.append( review_to_words( train["text"][i] ) )
        
        vectorizer = TfidfVectorizer(analyzer = "word",
            			     tokenizer = None,
            			     preprocessor = None)
        
        train_x = vectorizer.fit_transform(clean_train_data)
        train_x = train_x.toarray()
        print "shape: ", train_x.shape
        
        for task in tasks:
            start = timeit.default_timer()
            train_y = train[task]
            
            					   
            print "cross validation"
            print language, task, "#"*50
            
            if task not in ["gender", "age"]:
                clf = svm.SVR(kernel='linear', C=1)
                scoring_function = 'mean_squared_error'
            else:
                clf = svm.SVC(kernel='linear', C=1)
                scoring_function = 'accuracy'
            
            if not (language in ["dutch", "italian"] and task in ["age"]):
                scores = cross_validation.cross_val_score( clf, train_x, train_y, cv=n_folds, scoring=scoring_function)
                print "svc(kernel='linear', C=1) scores: "
                print "mean: ", scores.mean()
                print scores
            
                worksheet.write(index,0,language)
                worksheet.write(index,1,task)
                worksheet.write(index,2,scores.mean())
            stop = timeit.default_timer()
            time_elapsed = stop - start
            print time_elapsed
            index = index + 1
            
if __name__ == "__main__":
    main(sys.argv[1:])
        
        



