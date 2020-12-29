
from pathlib import Path
import random, sys, os
#from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
import yaml, pickle
import warnings
#from importlib import reload


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix, classification_report


# Surpress warnings due to different sklearn versions
# used for saving and loading
warnings.filterwarnings("ignore")

#filepath = Path.cwd() / 'data'     # wikipedia language identification data 2018 

# Import glossary for acronyms:
#--------------------------------
with open('glossary.yaml') as f:    
    glossary = yaml.load(f, Loader=yaml.FullLoader)

# Load the model from disk
#----------------------------
#filename = 'trained_model.pkl'
#loaded_model = pickle.load(open(filename, 'rb'))
#print('Model loaded.')

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
from flask import Flask, request, render_template

app = Flask(__name__)

#my_home_dir=os.environ['HOME'] 
#file_abs_path = os.path.join(my_home_dir,"utils.py")
#sys.path.append(file_abs_path)


def init():
    global loaded_model
    #model = joblib.load("model.h5")
    filename = 'trained_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    print('Model loaded.')


@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    filename = 'trained_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))

    # Predict
    #---------
    fore = loaded_model.predict(pd.Series([text]))[0]
    #print('Prediction: {}'.format(glossary['label_desc'].get(fore, 'Unknown')))
    #processed_text = 'You inserted: {}'.format(text)
    processed_text = 'Prediction: {}'.format(glossary['label_desc'].get(fore, 'Unknown'))
    return processed_text


if __name__ == "__main__":
    #init()
    #filename = 'trained_model.pkl'
    #loaded_model = pickle.load(open(filename, 'rb'))
    app.run(debug=True, host='0.0.0.0')

"""
from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
""" 
