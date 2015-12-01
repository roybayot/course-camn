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

def clean_text(raw_text):
    review_text = BeautifulSoup(raw_text).get_text()
    words = review_text.lower().split()
    return(" ".join(words))

def clean_all_text(allText, numLines):
    for i in xrange(0, numLines):
        
def makeTFIDF():

def main():
    datafile = "summary-english-file"

    train = pd.read_csv(datafile, header=0, delimiter="\t", quoting=1)


    cleanFile

    getNumImptWords()

    doSVMwithPoly()
    doSVMwithRBF()

    doRandomForest()
    doBoosting()

