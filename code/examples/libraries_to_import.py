import numpy as np
import re
import os
import pickle 
import pandas as pd
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from collections import defaultdict
import random 
import tomotopy as tp
import math
from functools import reduce
import operator
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from itertools import combinations


# custom
from utilities import create_data_words, create_corpora_voc,from_modeldiz_to_realdiz
from baptism_functions import *
from preprocess_functions import pre_preprocess_df,preprocess_df,remove_stopwords
from precisionrecallacc_functions import *
from perplexityUtilities import * 