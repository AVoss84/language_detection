
from pathlib import Path
import sys, os, warnings
import numpy as np
import pandas as pd
from copy import deepcopy
import yaml, pickle


# Surpress warnings due to different sklearn versions
# used for saving and loading
warnings.filterwarnings("ignore")

#filepath = Path.cwd() / 'data'     # wikipedia language identification data 2018 

# Import glossary for acronyms:
#--------------------------------
with open('glossary.yaml') as f:    
    glossary = yaml.load(f, Loader=yaml.FullLoader)

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

    # Load the model from disk
    #----------------------------
    filename = 'trained_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    #print('Model loaded.')

    # Predict
    fore = loaded_model.predict(pd.Series([text]))[0]
    #print('Prediction: {}'.format(glossary['label_desc'].get(fore, 'Unknown')))
    #processed_text = 'You inserted: {}'.format(text)
    processed_text = 'Identified language: {}'.format(glossary['label_desc'].get(fore, 'Unknown'))
    return processed_text


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

