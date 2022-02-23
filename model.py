#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
# from textblob import TextBlob
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[8]:


import nltk
nltk.download('wordnet')


# In[9]:


dataset_path = "ProjectDataset.csv"

df = pd.read_csv(dataset_path)
df.head()


# In[10]:


df['Organization'] = df['URL'].str.split('/')[:].str[4]

df['Organization'].unique()

df['EmployeeType'] = df.ReviewDetails.str.split('-')[:].str[0]

def EmployeeType(value):
  if 'Former' in value:
    return 'Former Employee'
  else:
    return 'Current Employee'

df['EmployeeType'] = df.apply(lambda row: EmployeeType(row['EmployeeType']), axis = 1)

df.head()


# In[11]:


df['Location'] = df.ReviewDetails.str.split('-')[:].str[1].replace(' ', '')

df.Location[df['Location'] == '   '] = 'Not Available'

df.drop(['URL', 'ReviewDetails'], axis = 1, inplace = True)

df['Review'] = df['ReviewTitle'] + ' ' + df['CompleteReview']

df.isnull().sum()

df = df.dropna()

df.drop(['ReviewTitle', 'CompleteReview'], axis = 1, inplace = True)

df.head()


# In[12]:


df = df[['Review', 'Organization', 'EmployeeType', 'Location', 'Rating']]
df.head()

df.isnull().sum()

df.shape

df.to_csv("RequiredDataset.csv")


# In[13]:


df.drop(['Organization', 'EmployeeType', 'Location'], axis = 1, inplace = True)

df.head()

def partition(x):
  if x >= 4:
    return 1
  else:
    return 0

label = df['Rating']
actual_label= label.map(partition)
df['Rating'] = actual_label

df.head()

df['Rating'].value_counts()


# In[14]:


def expand_contractions(review):

  #specific
  review = re.sub(r'won\'t', 'will not', review)
  review = re.sub(r"can\'t", "can not", review)

  #general
  review = re.sub(r"n\'t", " not", review)
  review = re.sub(r"\'re", " are", review)
  review = re.sub(r"\'s", " is", review)
  review = re.sub(r"\'d", " would", review)
  review = re.sub(r"\'ll", " will", review)
  review = re.sub(r"\'t", " not", review)
  review = re.sub(r"\'ve", " have", review)
  review = re.sub(r"\'m", " am", review)

  return review


# In[15]:


preprocessed_reviews = []

for review in tqdm(df['Review'].values):
  #lowercase
  review = review.lower()

  #removing numbers from review
  review = re.sub(r'[0-9]', '', review)

  #removing special characters from review
  review = re.sub(r'[^A-Za-z]', ' ', review)

  #removing extra spaces
  review = re.sub(' +', ' ', review)

  words = review.split()

  lemmatizer = WordNetLemmatizer()

  words = [lemmatizer.lemmatize(word) for word in words if not word in set(stopwords.words("english"))]

  review = ' '.join(words)

  preprocessed_reviews.append(review)


# In[18]:


preprocessed_reviews[0:10]

len(preprocessed_reviews)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='word', ngram_range=(2, 3), min_df = 10)
X = cv.fit_transform(preprocessed_reviews).toarray()
y = df.iloc[:, 1].values

pickle.dump(cv, open("transform.pkl", "wb"))

X.shape

# cv.get_feature_names_out()

y.shape


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)


# In[20]:


pickle.dump(classifier, open('model.pkl', 'wb'))

y_pred = classifier.predict(X_test)


# In[21]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
print("---- Scores ----")
print("Precision score is: {}".format(round(precision,2)))
print("Recall score is: {}".format(round(recall,2)))

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

conf_matrix


# In[ ]:




