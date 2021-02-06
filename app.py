from flask import Flask, request, render_template, redirect, url_for
#from pathlib import Path
import sys, os, warnings
import numpy as np
import pandas as pd
from copy import deepcopy
import yaml, pickle
#import requests
#from bs4 import BeautifulSoup

# Surpress warnings due to different sklearn versions
# used for saving and loading
warnings.filterwarnings("ignore")

# Import glossary for acronyms:
#--------------------------------
with open('glossary.yaml') as f:    
    glossary = yaml.load(f, Loader=yaml.FullLoader)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def hello_admin():
   return 'Hello Admin'


# User/client sends/posts form data via browser to webserver
# the text language is then classified:
#----------------------------------------------------------
@app.route('/', methods = ['POST', 'GET'])
def my_form_post():

    if request.method == "POST":
        text = request.form['text']
        print(request.method)
        #url = request.form['url']

        # Load the model from disk
        #----------------------------
        filename = 'trained_model.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
        #print('Model loaded.')

        # Predict
        #---------
        fore = loaded_model.predict(pd.Series([text]))[0]

        #class_probs = pd.DataFrame(trained.predict_proba(pd.Series([new_sent])), columns=trained.classes_, index=['Prob.']).T
        #class_probs.sort_values(by = 'Prob.', ascending=False).head(10)

        #print('Prediction: {}'.format(glossary['label_desc'].get(fore, 'Unknown')))
        #processed_text = 'You inserted: {}'.format(text)
        processed_text = '"{}"'.format(glossary['label_desc'].get(fore, 'Unknown'))
        #return processed_text
        #return redirect(url_for('hello_admin'))
    
        result = processed_text #{'my_prediction' : processed_text} 
        #if request.method == 'POST':
            #result = request.form
        return render_template("index.html", result = result)


"""
@app.route('/scrap', methods = ['POST', 'GET'])
def web_scrap():
    if request.method == "POST":
        # get url that the person has entered
        try:
            url = request.form['url']
            r = requests.get(url)
        except:
            errors.append(
                "Unable to get URL. Please make sure it's valid and try again."
            )
            return render_template('index.html', errors=errors)
    if r:
        # text processing
        raw = BeautifulSoup(r.text, 'html.parser').get_text()        
"""


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

