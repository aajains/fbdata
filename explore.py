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
## Filtering top words out: example
#words = word_tokenize(df["content"][1])
#stop_words = set(stopwords.words("English"))
#filtered_sentence = [w for w in words if not w in stop_words]
## Stemming example
#ps = PorterStemmer()
#for w in word_tokenize(df["content"][1]):
#    print(ps.stem(w))
## Part of speech tagging example
## Training
#train_text = df["content"][1]#state_union.raw("2005-GWBush.txt")
#custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
## Using the trained tokenizer
#sample_text = df["content"][1]#state_union.raw("2006-GWBush.txt")
#tokenized = custom_sent_tokenizer.tokenize(sample_text)
#def process_content():
#    try:
#        for i in tokenized:
#            words = word_tokenize(i)
#            tagged = nltk.pos_tag(words)
#            # Chunking
#            #chunkGram = r"""Chunk:{<RB.?>*<DT>+} """
#            # Chinking
#            #chunkGram = r"""Chunk:{<.*>+} 
#            #                    }<VB.? | IN | DT>+{"""
#            #chunkParser = nltk.RegexpParser(chunkGram)
#            #chunked = chunkParser.parse(tagged)
#            #chunked.draw()
#            #print(chunked[0])
#            #print(chunked)
#            namedEnt = nltk.ne_chunk(tagged, binary=True)
#            namedEnt.draw()
#            
#    except Exception as e:
#        print(str(e))
#process_content()
## Lemmatization example:
#lemmatizer = WordNetLemmatizer()
#print({w:lemmatizer.lemmatize(w) for w in word_tokenize(df["content"][1])})
## Wordnet examples
#syns = wordnet.synsets("program")
#print(syns[0].name())
#print(syns[2].lemmas()[0].name())
#print(syns[0].definition())
#print(syns[0].examples())
#synonyms = []
#antonyms = []
#for syn in wordnet.synsets("good"):
#    for lem in syn.lemmas():
#        synonyms.append(lem.name())
#        if lem.antonyms():
#            antonyms.append(lem.antonyms()[0].name())
#            
#print(set(synonyms))
#print(set(antonyms))
# Similarity example using wordnet
w1 = wordnet.synset("coffee.n.01")
w2 = wordnet.synset("mug.v.01")
print(w1.wup_similarity(w2))



























