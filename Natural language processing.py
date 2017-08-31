# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Clearing the texts
import re
import nltk
# downloading  the stopwords 
nltk.download('stopwords')
# as it is been downloaded in the nltk so just importing it from nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
#going to the whole dataset and applying cleaning 
# this is sub is used to just include words which is neceessary as 
 # as here i have only choosed a-z and A,Z alphabets only
 # changing every alphabets into lower case  
 # making a list and split every words and taking as a list
 # Porter Stemmer here stems the data which find the root of the words
 # such loved is the pas tense of the love so it replace the loved with love 
  # here each applying the stopwords checking not in stop words and if it is in it so 
  # removing it
 # creating an array and appending the new cleaning list of string 
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#  Creating the Bag of Words model 
# creating an object of an count vectorizer
# set of dependent variable and independent variables
# y as same used in the classification #
# created two different dataset X and y
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the datset into training and testing set 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting the dataset into the one of the classification 
# algorithm such as Naves bayes as we can also apply decision tree
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test of the results 
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix 
# so that how the many of the values is true positive or true negave 
# using the method of confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# print the accuracy 
print("Accuracy of the predited values are ",(55+91/200))

  
