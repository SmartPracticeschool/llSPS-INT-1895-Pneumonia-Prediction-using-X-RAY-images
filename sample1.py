# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:13:49 2020

@author: Sai Nidhi
"""

from keras.models import load_model
import pickle 

with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cls = load_model('twitter1.h5')
cls.compile(optimizer='adam',loss='binary_crossentropy')

x_intent="I am a bad girl"
x_intent=cv.transform([x_intent])
y_pred=cls.predict(x_intent)
y_pred=(y_pred>0.5)
print(y_pred)
