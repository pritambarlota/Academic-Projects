# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:20:10 2018

@author: prita
"""

# Naive Bayes

# Importing the libraries
import sys
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:49:47 2018

@author: prita
"""
## Step 1 - Data collection : Download the data
sms_raw = pd.read_csv("C:\PritamData\Stats_With_R\Homework\sms_spam.csv")

## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nNo of rows and columns\n", sms_raw.shape)               ## No of rows and columns
print("\nDatatypes \n", sms_raw.dtypes)              ## Structure
print("\nCount of Spam\n",sms_raw.groupby('type').size())       ## count

print("\nLenght ", len(sms_raw.type))

print("\n Print one line - ".join(sms_raw.text[0].split("\n")))
   
        
X = sms_raw.iloc[:,1:]
y = sms_raw.iloc[:,0]

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#print("\nSize of Training Dataset\n",X_train[0].size())       ## count for X_train
print("\nSize of Training Dataset\n",y_train.count())       ## count for y_train
print("\nSize of Testing Dataset\n",y_test.count())       ## count for y_test


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.text)
print("Dataset after CountVectorizer :", X_train_counts.shape)
df1 = pd.DataFrame(X_train_counts.toarray(),columns=count_vect.get_feature_names())
print("Datset\n", df1.head)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("Dataset after TfidfTransformer :", X_train_tfidf.shape)
df2 = pd.DataFrame(X_train_tfidf.toarray())
print("Datset\n", df2)

## Step 3: Model Training:  Training a model on the data ----
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB().fit(X_train_tfidf, y_train)
print("Classifier :", classifier)

#clf = GaussianNB().fit(X_train_tfidf, y_train)

## Step 4: Model Evaluation : Evaluating model performance ----
X_test_counts = count_vect.transform(X_test.text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

t0=time()
y_pred = classifier.predict(X_test_tfidf)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: ",classification_report(y_test, y_pred))

## Accuracy using classifier function
print("Training Accuracy : ", classifier.score(X_train_tfidf, y_train, sample_weight=None))
print("Testing Accuracy : ", classifier.score(X_test_tfidf, y_test, sample_weight=None))

## Step 5: Model Improvement : Improving model performance ----




