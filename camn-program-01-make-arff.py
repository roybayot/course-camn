#!/usr/bin/python

"""
This program makes an arff file from the text files.
Headers would be words. Numbers would be tfidf.

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

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

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
    languages = ["english"]
    datafiles = ["summary-english-truth.txt"]
    tasks = ["gender",
             "age"]

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
            makeArff(train_y, train_x, task)
if __name__ == "__main__":
    main(sys.argv[1:])
        
        



