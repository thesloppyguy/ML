from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer  # removes the tens of the words
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('DATA/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# while gettting the data be mindfull of "" '' as theyb are also part of the sentence and can be harmfull to the system
# quoting = 3 helps with thi stype of issue

# cleaning the text
nltk.download('stopwords')  # useless words like 'the and a is it if'

corpus = []  # contains cleaned text

for i in range(0, 1000):
    # remove special characters
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # make everything lowercase
    review = review.lower()
    # split the sentence into words
    review = review.split()
    # removing tens and stopwords form the list
    ps = PorterStemmer()
    stopwords_list = stopwords.words('english')
    # as we have deduced we need not in our training dataset we remove it from the set of words to be removed
    stopwords_list.remove('not')

    review = [ps.stem(item)
              for item in review if not item in set(stopwords_list)]
    review = ' '.join(review)  # joins all the words with ' ' in between them
    corpus.append(review)

# making the bag of words - vector of 20k words assigned to each index
# 1500 most frequent word should only stay in the vector as 1566 is the number of words and its just a round off
cv = CountVectorizer(max_features=1500)
# needs to be a 2d array for classification
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# apply classifiction

rfc = RandomForestClassifier(
    n_estimators=100, criterion="entropy", random_state=0)
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

ac = accuracy_score(y_test, y_pred)
print(ac)
