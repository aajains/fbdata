#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:07:14 2017

@author: aashishjain
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:54:59 2017

@author: aashishjain
"""
import pandas as pd
import os
import operator
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
#from pylab import *
# Getting current working dir
mydir = os.getcwd()
df = pd.read_csv(mydir+"/bbc-228735667216.csv", encoding='utf-16')
nrows = len(df)
# Getting fractions of nulls in each column in a dictionary
frac_nulls = {col: (sum(df[col].isnull())/nrows) for col in df.columns}
frac_nulls = sorted(frac_nulls.items(), key = operator.itemgetter(1))
# Create a column for post_id
df["post_id"] = df["id"].str.split("_").str[1].str.split("\"").str[0]
# Change date column format to datetime
df["posted_at"] = pd.to_datetime(df["posted_at"])
# Create a new column for the text content
df['content'] = df[['message', 'description']].bfill(axis = 1)["message"]
df["content"] = df["content"].replace(" ",float("nan"))
df['content'] = df[['content', 'name']].bfill(axis = 1)["content"]
# Following rows have no text content in them
#df["content"][pd.isnull(df["content"])]
#5030     NaN
#11620    NaN
#14191    NaN
#17471    NaN
#18001    NaN
#18411    NaN
#18556    NaN
#18686    NaN
#18752    NaN
