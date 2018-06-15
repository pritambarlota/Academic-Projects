# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:38:55 2018

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
from sklearn.neural_network import MLPClassifier

## Example: Estimating Wine Quality ----
## Step 1 - Data collection : Download the data
wine = pd.read_csv("C:\PritamData\Stats_With_R\Homework\wine.csv")
wine.head(5)

## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nNo of rows and columns\n", wine.shape)               ## No of rows and columns
print("\nDatatypes \n", wine.dtypes)              ## Structure

X = wine.iloc[:,1:12]
y = wine.iloc[:,0]

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("\nSize of Training Dataset\n",X_train.shape)     ## count for X_train
print("\nSize of Testing Dataset\n",X_test.shape)     ## count for X_test

print("\nSize of Training Dataset\n",y_train.count())       ## count for y_train
print("\nSize of Testing Dataset\n",y_test.count()) 

# summary statistics of the wine data
print("\nSummary of Numeric Variables\n",wine.describe(include=[np.number]))    
     
print("\nCheck any Null values :\n", X.isnull().sum())
print("\nCheck if all Finite values :\n", np.isfinite(X).sum())
print("\nCheck any NaN values :\n", np.isnan(X).sum())

## Step 3: Model Training:  Training a model on the data ----
### Using MLPRegressor from nueral net
nnet = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

nnet.fit(X_train, y_train)
print("\n\nMLPClassifier :", nnet)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = nnet.predict(X_test)
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
print(nnet.score(X_train, y_train, sample_weight=None))
print("\nTesting Accuracy : ")
print(nnet.score(X_test, y_test, sample_weight=None))  
 
## Step 5: Model Improvement : Improving model performance ----
### Using MLPRegressor from nueral net
nnet = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(15,15), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
nnet.fit(X_train, y_train)
print("\n\nMLPClassifier :", nnet)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = nnet.predict(X_test)
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
print(nnet.score(X_train, y_train, sample_weight=None))
print("\nTesting Accuracy : ")
print(nnet.score(X_test, y_test, sample_weight=None))  
  

### Using MLPRegressor from nueral net
nnet = MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(200,200), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
nnet.fit(X_train, y_train)
print("\n\nMLPRegressor using 100, 100 nodes and layers :", nnet)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = nnet.predict(X_test)
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
print(nnet.score(X_train, y_train, sample_weight=None))
print("\nTesting Accuracy : ")
print(nnet.score(X_test, y_test, sample_weight=None))  
  
