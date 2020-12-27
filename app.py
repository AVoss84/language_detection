
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
#---------
fore = loaded_model.predict(pd.Series([new_sent]))[0]
print('Prediction: {}'.format(glossary['label_desc'].get(fore, 'Unknown')))







