

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
from textblob import TextBlob
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import nltk
nltk.download('wordnet')

dataset_path = "ProjectDataset.csv"

df = pd.read_csv(dataset_path)
df.head()

# df.shape

# df.columns

# df.info()

# df.isnull().sum()

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

df['Location'] = df.ReviewDetails.str.split('-')[:].str[1].replace(' ', '')

df.Location[df['Location'] == '   '] = 'Not Available'

df.head()

df.drop(['URL', 'ReviewDetails'], axis = 1, inplace = True)

df['Review'] = df['ReviewTitle'] + ' ' + df['CompleteReview']

df.isnull().sum()

df = df.dropna()

df.head()

df.drop(['ReviewTitle', 'CompleteReview'], axis = 1, inplace = True)

df.head()

"""Rearranging the columns in the Dataframe"""

df = df[['Review', 'Organization', 'EmployeeType', 'Location', 'Rating']]
df.head()

df.isnull().sum()

df.shape

df.to_csv("/gdrive/MyDrive/RequiredDataset.csv")

"""**EDA**"""

sns.countplot(x = 'Rating', data = df)

sns.countplot(x = 'EmployeeType', data = df)

plt.figure(figsize=(25,8))

sns.countplot(x = 'Organization', data = df)

plt.xticks(rotation = 90)

!pip install pandasql
from pandasql import sqldf

first_10 = sqldf('''
SELECT *
FROM df
WHERE Organization in (SELECT Organization FROM df GROUP BY Organization ORDER BY count(Organization) DESC LIMIT 10)
''')

plt.figure(figsize=(20,8))

sns.countplot(x = 'Organization', hue = 'Rating', data = first_10 )

plt.xticks(rotation = 90)

last_10 = sqldf('''
SELECT *
FROM df
WHERE Organization in (SELECT Organization FROM df GROUP BY Organization ORDER BY count(Organization) ASC LIMIT 10)
''')

plt.figure(figsize=(20,8))

sns.countplot(x = 'Organization', hue = 'Rating', data = last_10)

plt.xticks(rotation = 90)

"""**Text Preprocessing**

Cleaning text data and preprocessing it is one of the important steps in any Machine Learning/Deep Learning Project. By preprocessing the data we will be removing all the unnecesary information which does not add value to the model, this helps in improving the performace of the model.

Text Preprocessing includes:
1. Lowercase the text data
2. Remove puntuations and special characters
3. Remove numbers and alphanumeric words
4. Remove html tags if present
5. Removes any extra spaces
6. Expand contractions
7. Remove stop words
8. Tokenization
9. Stemming
10. Lemmatization

Note: Preprocessing includes additional steps depending on the data we have, and how well we can process it.
"""

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

# https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

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

preprocessed_reviews[0:10]

len(preprocessed_reviews)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='word', ngram_range=(2, 3), min_df = 10)
X = cv.fit_transform(preprocessed_reviews).toarray()
y = df.iloc[:, 1].values

pickle.dump(cv, open("/gdrive/MyDrive/transform.pkl", "wb"))

X.shape

cv.get_feature_names_out()

y.shape

"""**Model Building**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)

pickle.dump(classifier, open('/gdrive/MyDrive/model.pkl', 'wb'))

y_pred = classifier.predict(X_test)

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

"""**Predictions**"""

def predict_sentiment(sample_review):

  expand_contractions(sample_review)
  #lowercase
  sample_review = sample_review.lower()

  #removing numbers from review
  sample_review = re.sub(r'[0-9]', '', sample_review)

  #removing special characters from review
  sample_review = re.sub(r'[^A-Za-z]', ' ', sample_review)

  #removing extra spaces
  sample_review = re.sub(' +', ' ', sample_review)

  words = sample_review.split()

  lemmatizer = WordNetLemmatizer()

  words = [lemmatizer.lemmatize(word) for word in words if not word in set(stopwords.words('english'))]

  sample_review = ' '.join(words)

  temp = cv.transform([sample_review]).toarray()

  return classifier.predict(temp)

sample_review = "This company is not good to work for. It is not helpful for employee growth and worst work environment."

if predict_sentiment(sample_review):
  print('This is a POSITIVE review.')
else:
  print('Oops! This is a NEGATIVE review!')

sample_review = "Best company to work with for freshers especially."

if predict_sentiment(sample_review):
  print('This is a POSITIVE review.')
else:
  print('Oops! This is a NEGATIVE review!')
