
from pathlib import Path
import random, sys, os
#from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
from importlib import reload
import yaml, pickle
#from utils import read, clean_text
import utils

reload(utils)

current_dir = Path.cwd()
#home_dir = Path.home()
#files = os.listdir(os.curdir)

filepath = Path.cwd() / 'data'     # wikipedia language identification data 2018 

# Load data:
#-------------
x_train = read(filepath / 'x_train.txt')
x_test = read(filepath / 'x_test.txt')
y_train = read(filepath / 'y_train.txt')
y_test = read(filepath / 'y_test.txt')

# Format data a bit:
train = pd.DataFrame({'text': x_train, 'label' :y_train})
train.label = train.label.str.rstrip('\n')
test = pd.DataFrame({'text': x_test, 'label' :y_test})
test.label = test.label.str.rstrip('\n')      # remove \n characters

# Import glossary for acronyms:
#--------------------------------
with open('glossary.yaml') as f:    
    glossary = yaml.load(f, Loader=yaml.FullLoader)


# Filter languages to be trained on:
filter_list = ['deu', 'fra', 'eng', 'est', 'fin', 'nld', 'rus', 'swe', 'zho', 'tha', 
               #'est', 'ukr', 'mya'
               'tur', #'vie',
               'heb', 'hrv', 'ind', 'lat',
               #'kor', 'mal', 
               'nno', 'pol', 'por', 'spa', 'sqi', 'srp', 'zh-yue',
               'slk','dan', 'ces', 'ron','ita', 'hun', 'bul', 'ara', 'jpn']

train_set = train[train.label.isin(filter_list)]
test_set = test[test.label.isin(filter_list)]

#train_set = deepcopy(train)
#test_set = deepcopy(test)

print(train_set.shape)
print(test_set.shape)
#print(any(train.label.isin(filter_list)))
#print(any(test.label.isin(filter_list)))


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix, classification_report


pipeline = Pipeline([
   ('cleaner', utils.clean_text(verbose=False)),
   ('vectorizer', CountVectorizer(max_features=None, lowercase=True, 
                       token_pattern = '(?u)(?:(?!\d)\w)+\\w+', analyzer = 'char',  
                       ngram_range=(2, 2), tokenizer = None, stop_words = None)),  
   #('scaler', StandardScaler(with_mean=False)),
   ('model', MultinomialNB())
])

# Train:
trained = pipeline.fit(train_set['text'], train_set['label'])

#post_prob = pipeline.predict_proba(train_set['text'])

#-------------
# Save model:
#-------------
filename = 'trained_model.pkl'
pickle.dump(pipeline, open(filename, 'wb'))
print('Model saved!')

# Predict:
y_pred = pipeline.predict(test_set['text'])

#print(confusion_matrix(test_set['label'], y_pred))
print(classification_report(test_set['label'], y_pred))

#----------------
# Test sentence:
#----------------
new_sent = "Gestern abend ging ich in ein Restaurant. da saß ein Typ neben mir und meinte nur!"   # german
new_sent = "Yesterday I was in the city. The weather was super nice."     # eng
new_sent = "Вчера был в городе. Погода была очень хорошей."       # russian
new_sent = "Ayer estuve en la ciudad. El clima estuvo super agradable."   # spanish
new_sent = "Hier, j'étais en ville. Le temps était super beau."      # french
new_sent = "Heri eram in civitate. Super tempestas est delicatus."     # latin
new_sent = "Ieri ero in città. Il tempo è stato bellissimo."     # italian
new_sent = "Igår var jag i staden. Vädret var supertrevligt."    # sweden
new_sent = "ฉันอยากบอกสามีของฉันว่าฉันอยากได้แหวนเพชร ฮี่ฮี่ฮี่"     # if not working check result of preprocessing!!

# Predict
fore = pipeline.predict(pd.Series([new_sent]))[0]
print('Prediction: {}'.format(glossary['label_desc'].get(fore, 'Unknown')))

#clt = clean_text(verbose=False)
#clt.fit_transform(pd.Series([new_sent]))
    
