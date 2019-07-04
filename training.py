# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:12:00 2019

@author: Mohit Bansal
"""

import json
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB ,GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer


with open('Sarcasm_Headlines_Dataset.json') as read_file:
    jsondata=read_file.readlines()

headlines=[]
labels=[]
for i in jsondata:
    x=json.loads(i)
    headlines.append(x["headline"])
    labels.append(x["is_sarcastic"])
headlines=np.array(headlines)
labels=np.array(labels)

import re
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, len(headlines)):
    review = re.sub('[^a-zA-Z]', ' ', headlines[i])
    review = review.lower()
    review = review.split()

    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

tf = TfidfVectorizer(min_df = 1,stop_words='english')
features = tf.fit_transform(corpus).toarray()

================================                                  
bnb=BernoulliNB()
bnb.fit(features,labels)
labels_pred = bnb.predict(feature_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = bnb, X = features, y = labels, cv = 10)
print ("mean accuracy is",accuracies.mean())
print (accuracies.std())
============================================
from sklearn.externals import joblib
joblib.dump(bnb,'bnb.pkl')
joblib.dump(tf,'tf.pkl')
                                                                  