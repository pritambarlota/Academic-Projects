# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:30:33 2018

@author: prita
"""

import sys
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.svm import SVC

##### Support Vector Machines -------------------
## Example: Optical Character Recognition ----

## Step 2: Exploring and preparing the data ----
# read in data and examine structure
letters = pd.read_csv("C:\PritamData\Stats_With_R\Homework\letterdata.csv")
letters.head(5)

## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nNo of rows and columns\n", letters.shape)               ## No of rows and columns
print("\nDatatypes \n", letters.dtypes) 

# divide into training and test data
X = letters.iloc[:, 1:17]
y  = letters.iloc[:, 0 ]

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("\nSize of Training Dataset\n",X_train.shape)     ## count for X_train
print("\nSize of Testing Dataset\n",X_test.shape)     ## count for X_test

print("\nSize of Training Dataset\n",y_train.count())       ## count for y_train
print("\nSize of Testing Dataset\n",y_test.count()) 

# summary statistics of the wine data
print("\nSummary of Numeric Variables\n",X.describe(include=[np.number]))   

## Step 3: Training a model on the data ----
# begin by training a simple linear SVM
clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

# look at basic information about the model
clf.fit(X_train, y_train)
print("\n\n SVM  :", clf)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = clf.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")

print("\nConfusion matrix after prediction")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nAccuracy %: ")
print(accuracy_score(y_test, y_pred)*100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))                                                  #(tn, fp, fn, tp)

#The kappa statistic adjusts accuracy by accounting for the possibility of a correct prediction by chance alone.
print("\nKappa Score:")
print(cohen_kappa_score(y_test, y_pred))

## Accuracy using classifier function
print("\nTraining Accuracy : ")
print(clf.score(X_train, y_train, sample_weight=None))
print("\nTesting Accuracy : ")
print(clf.score(X_test, y_test, sample_weight=None))


## Step 5: Improving model performance ----

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

# look at basic information about the model
clf.fit(X_train, y_train)
print("\n\n SVM  :", clf)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = clf.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")

print("\nConfusion matrix after prediction")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nAccuracy %: ")
print(accuracy_score(y_test, y_pred)*100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))                                                  #(tn, fp, fn, tp)

#The kappa statistic adjusts accuracy by accounting for the possibility of a correct prediction by chance alone.
print("\nKappa Score:")
print(cohen_kappa_score(y_test, y_pred))

## Accuracy using classifier function
print("\nTraining Accuracy : ")
print(clf.score(X_train, y_train, sample_weight=None))
print("\nTesting Accuracy : ")
print(clf.score(X_test, y_test, sample_weight=None))