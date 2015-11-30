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



