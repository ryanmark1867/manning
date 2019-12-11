# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/

# this is the Python action file for streetcar delay prediction


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message("Hello World!")
#
#         return []

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

# main code block

# 'https://raw.githubusercontent.com/ryanmark1867/chatbot/master/datasets/links_small.csv'
# "https://github.com/ryanmark1867/chatbot/raw/master/images/thumb_up.jpg"
# https://github.com/ryanmark1867/manning/blob/master/models/scmodeldec1.h5

# model_path = "https://github.com/ryanmark1867/manning/raw/master/models/scmodeldec1.h5"
model_path = 'C:\personal\chatbot_july_2019\keras_models\scmodeldec1.h5'

loaded_model = load_model(model_path)
loaded_model.summary()
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

#
preds = loaded_model.predict(score_sample, batch_size=BATCH_SIZE)
print("pred is ",preds)

print("preds[0] is ",preds[0])
print("preds[0][0] is ",preds[0][0])


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
