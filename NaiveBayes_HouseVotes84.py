# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:10:27 2018

@author: prita
"""

import sys
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB

## Step 1 - Data collection : Download the data
HouseVotes = pd.read_csv("C:\PritamData\Stats_With_R\Homework\HouseVotes84.csv")

## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nNo of rows and columns\n", HouseVotes.shape)               ## No of rows and columns
print("\nDatatypes \n", HouseVotes.dtypes)              ## Structure
print("\nCount of Spam\n",HouseVotes.groupby('Class').size())       ## count
print("\nLenght ", len(HouseVotes.Class))

X = HouseVotes.iloc[:,2:]
y = HouseVotes.iloc[:,1]

## Step 2: Data exploration and preparation : Feature Scaling 

#for i in X_train.columns.values:
X = X.replace({'V1': {'y': 1, 'n': 0}})
X = X.replace({'V2': {'y': 1, 'n': 0}})
X = X.replace({'V3': {'y': 1, 'n': 0}})
X = X.replace({'V4': {'y': 1, 'n': 0}})
X = X.replace({'V5': {'y': 1, 'n': 0}})
X = X.replace({'V6': {'y': 1, 'n': 0}})
X = X.replace({'V7': {'y': 1, 'n': 0}})
X = X.replace({'V8': {'y': 1, 'n': 0}})
X = X.replace({'V9': {'y': 1, 'n': 0}})
X = X.replace({'V10': {'y': 1, 'n': 0}})
X = X.replace({'V11': {'y': 1, 'n': 0}})
X = X.replace({'V12': {'y': 1, 'n': 0}})
X = X.replace({'V13': {'y': 1, 'n': 0}})
X = X.replace({'V14': {'y': 1, 'n': 0}})   
X = X.replace({'V15': {'y': 1, 'n': 0}})
X = X.replace({'V16': {'y': 1, 'n': 0}}) 

##Fill every column with its own most frequent value
X = X.apply(lambda x:x.fillna(x.value_counts().index[0]))

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.23, random_state = 0)

print("\nSize of Training Dataset\n",y_train.count())       ## count for y_train
print("\nSize of Testing Dataset\n",y_test.count())       ## count for y_test
print("\nProportion of democrat - Train\n",y_train.value_counts() / 334 *100)       ## count
print("\nProportion of republican - Test \n",y_test.value_counts() / 101 * 100)       ## count

## Step 3: Model Training:  Training a model on the data ----
# Fitting Naive Bayes to the Training set
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: ", classification_report(y_test, y_pred))

## Step 5: Model Improvement : Improving model performance ----
