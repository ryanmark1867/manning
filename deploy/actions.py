# This files contains the custom actions related to the Rasa model for the streetcar delay prediction project

# common import block
# common imports
from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher



import zipfile
import pandas as pd
import numpy as np
import time
# import datetime, timedelta
import datetime
from datetime import datetime, timedelta
from datetime import date
from dateutil import relativedelta
from io import StringIO
import pandas as pd
import pickle
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
# DSX code to import uploaded documents
from io import StringIO
import requests
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import math
from subprocess import check_output
#model libraries
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pickle import load
from custom_classes import encode_categorical
from custom_classes import prep_for_keras_input
from custom_classes import fill_empty
from custom_classes import encode_text
from datetime import date
from datetime import datetime
import logging

# main code block

logging.getLogger().setLevel(logging.WARNING)
logging.warning("logging check")

# hardcoded paths for model and pipeline files

# model_path = "https://github.com/ryanmark1867/manning/raw/master/models/scmodeldec1.h5"
model_path = 'C:\personal\chatbot_july_2019\keras_models\scmodeldec27b_5.h5'
pipeline2_path = 'C:\personal\chatbot_july_2019\streetcar_1\pipelines\sc_delay_pipeline_keras_prep_dec27b.pkl'

pipeline1_path = 'C:\personal\chatbot_july_2019\streetcar_1\pipelines\sc_delay_pipeline_dec27b.pkl'
prepped_data_path = 'C:\personal\chatbot_july_2019\streetcar_1\input_sample_data\prepped_dec27.csv'

loaded_model = load_model(model_path)
loaded_model.summary()
pipeline1 = load(open(pipeline1_path, 'rb'))
pipeline2 = load(open(pipeline2_path, 'rb'))
BATCH_SIZE = 1000

# brute force a scoring sample, bagged from test set
score_sample = {}
score_sample['hour'] = np.array([18])
score_sample['Route'] = np.array([0])
score_sample['daym'] = np.array([21])
score_sample['month'] = np.array([0])
score_sample['year'] = np.array([5])
score_sample['Direction'] = np.array([1])
score_sample['day'] = np.array([1])
# X_test  {'hour': array([18,  4, 11, ...,  2, 23, 17]), 'Route': array([ 0, 12,  2, ..., 10, 12,  2]), 'daym': array([21, 16, 10, ..., 12, 26,  6]),
 #   'month': array([0, 1, 0, ..., 6, 2, 1]), 'year': array([5, 2, 3, ..., 1, 4, 3]), 'Direction': array([1, 1, 4, ..., 2, 3, 0]),
 #  'day': array([1, 2, 2, ..., 0, 1, 1])}

'''
>>> from datetime import datetime

>>> now = datetime.now()
>>> now.day
28
>>> now.hour
15
>>> now.month
12
>>> now.weekday()
5
>>> 


'''


# dictionary of default values

score_default['hour'] = 12
score_default['Route'] = '501'
score_default['daym'] = 1
score_default['month'] = 1
score_default['year'] = '2019'
score_default['Direction'] = 'e'
score_default['day'] = 2

preds = loaded_model.predict(score_sample, batch_size=BATCH_SIZE)
logging.warning("pred is "+str(preds))

logging.warning("preds[0] is "+str(preds[0]))
logging.warning("preds[0][0] is "+str(preds[0][0]))

# example using pipeline on prepped data
# routedirection_frame = pd.read_csv(path+"routedirection.csv")
prepped_pd = pd.read_csv(prepped_data_path)
logging.warning("prepped_pd is"+str(prepped_pd))
prepped_xform1 = pipeline1.transform(prepped_pd)
prepped_xform2 = pipeline2.transform(prepped_xform1)
logging.warning("prepped_xform2 is"+str(prepped_xform2))

pred2 = loaded_model.predict(prepped_xform2, batch_size=BATCH_SIZE)
logging.warning("pred2 is "+str(pred2))

logging.warning("pred2[0] is "+str(pred2[0]))
logging.warning("pred2[0][0] is "+str(pred2[0][0]))


class ActionPredictDelay(Action):

    def name(self) -> Text:
        return "action_predict_delay"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("in action_predict_delay")
        if preds[0][0] >= 0.5:
            predict_string = "yes"
        else:
            predict_string = "no"
        dispatcher.utter_message("Delay prediction is:"+predict_string)

        return []


class ActionPredictDelayComplete(Action):
    ''' predict delay when the user has provided sufficient content to make a prediction'''
    def name(self) -> Text:
        return "action_predict_delay"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.warning("in action_predict_delay_complete")
        now = datetime.now()
        score_default['daym'] = now.day
        score_default['month'] = now.month
        score_default['year'] = now.year
        score_default['day'] = now.weekday()
        if preds[0][0] >= 0.5:
            predict_string = "yes"
        else:
            predict_string = "no"
        dispatcher.utter_message("Delay prediction is:"+predict_string)

        return []
