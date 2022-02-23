#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pickle
from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[8]:



model = pickle.load(open('model.pkl', 'rb'))
cv= pickle.load(open('transform.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():

  if request.method == 'POST':
        msg = request.form['message']
        data = [msg]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
  return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run()
    


# In[ ]:




