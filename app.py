
from pathlib import Path
import random, sys, os
#from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
import yaml, pickle
import warnings
#from importlib import reload
from utils import read, clean_text

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
filename = 'trained_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
print('Model loaded.')

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    # Predict
    #---------
    fore = loaded_model.predict(pd.Series([text]))[0]
    print('Prediction: {}'.format(glossary['label_desc'].get(fore, 'Unknown')))
    #processed_text = 'You inserted: {}'.format(text)
    processed_text = 'Prediction: {}'.format(glossary['label_desc'].get(fore, 'Unknown'))
    return processed_text


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
