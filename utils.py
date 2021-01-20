import random, sys, os, re, nltk, string, tarfile
#from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
#from gensim.parsing.preprocessing import strip_short
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from copy import deepcopy

nltk.download('punkt')
nltk.download('stopwords')
 
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


class preprocess_text(BaseEstimator, TransformerMixin):    
   """
   This is an extension of the clean_text class below. 
   Stemming and stopwords filtering were added for preprocessing.
   """ 
   def __init__(self, input_col, output_col, stemming = True, unique_tokens = True):
      self.name = output_col
      self.input_col = input_col
      self.stemming = stemming
      self.unique_tokens = unique_tokens
      if self.unique_tokens : print("Deduplicating tokens per document/sentence.")
      self.regexp = re.compile('(?u)(?:(?!\d)\w)+\\w+')        # remove punctuation and digits
      if self.stemming : self.stemmer = PorterStemmer() ; print("Applying Porter stemming.")
      self.stop_words = set(stopwords.words('english'))

   def fit(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      return self

   def transform(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      self.df_raw = deepcopy(X)
      d = self.df_raw[self.input_col].tolist()
      for i in range(len(d)):
        d[i] = " ".join(self.regexp.findall(str(d[i]))).lower().strip()        # strip ws and convert to lower
        tokens = word_tokenize(d[i])
        if self.unique_tokens:
            # Make tokens unique in sentence
            #tokens = list(set(tokens))   # this loses word order
            my_unique_sentence = []
            e = [my_unique_sentence.append(tok) for tok in tokens if tok not in my_unique_sentence]
            tokens = my_unique_sentence
 
        # Stem words and remove stopwords:
        if self.stemming:
           filtered_tokens = [self.stemmer.stem(w) for w in tokens if not w in self.stop_words]
        else:
           filtered_tokens = [w for w in tokens if not w in self.stop_words]
        d[i] = " ".join(filtered_tokens)
      self.df_raw[self.name] = d 
      return self.df_raw


class _clean_text(BaseEstimator, TransformerMixin):
    
   def __init__(self, input_col, output_col):
      self.name = output_col
      self.input_col = input_col
      self.regexp = re.compile('(?u)(?:(?!\d)\w)+\\w+')        # remove punctuation and digits

   def fit(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      return self

   def transform(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      self.df_raw = deepcopy(X)
      d = self.df_raw[self.input_col].tolist()
      for i in range(len(d)):
        d[i] = " ".join(self.regexp.findall(str(d[i]))).lower().strip()        # strip ws and convert to lower
      self.df_raw[self.name] = d 
      return self.df_raw
    

class StringCleaner(TransformerMixin):

    def __init__(self, mapping = {},  replace = True, verbose = True):
        # example mapping ={'column_name':{'find':'replacement}}
        self.replace = replace
        #self.convert_cols = convert_cols
        self.mapping = mapping
        self.verbose = verbose
    def fit(self):
        pass
    
    def transform(self, X, minimum_length = 5):
         if isinstance(X, pd.Series):
            return X
         if isinstance(X, pd.DataFrame):
            X_temp = deepcopy(X)
            for col in self.mapping:
                X_temp[col] = X_temp[col].str.lower()
              
            for col in self.mapping:
                for word in self.mapping[col]:
                    X_temp[col] = X_temp[col].str.replace(word,self.mapping[col][word])
            if self.verbose:
                print("Replaced strings in columns: "+ str([col for col in self.mapping]))
            return X_temp
                
    