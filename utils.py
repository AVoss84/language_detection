import random, sys, os, re, nltk, string, tarfile
#from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy

#nltk.download('punkt')
 
def read(file):
    '''Returns contents of a file'''
    with open(file, 'r', errors='ignore') as f:
        text = f.readlines()
    print('{} line read'.format(len(text)))    
    return text


class clean_text(BaseEstimator, TransformerMixin):
    """Clean text corpus for model training"""
    def __init__(self, verbose : bool = True):
        self.verbose = verbose

    def fit(self, X, y=None):
        return self    
    
    def transform(self, X):    
        corpus = deepcopy(X)
        cleaned_text = []
        # Preprocess:
        for z, se in enumerate(corpus.tolist()):
            if (z % 1000 == 0) & self.verbose: print('Processing document {}'.format(z))
            tokens = word_tokenize(se)
            # convert to lower case
            tokens = [w.lower() for w in tokens]
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            #nonPunct = re.compile('.*[A-Za-z].*')    # remove punctuation
            #raw_words = [w for w in tokens if nonPunct.match(w)]
            # remove remaining tokens that are not alphabetic
            #words = [word for word in stripped if word.isalpha()]    
            #words = [re.sub(r'\s+',' ',word) for word in stripped]
            #words = [word.strip() for word in stripped]
            # filter out stop words
            #stop_words = set(stopwords.words('english'))
            #words = [w for w in words if not w in stop_words]
            cleaned_text.append(' '.join(stripped))
        return cleaned_text  
