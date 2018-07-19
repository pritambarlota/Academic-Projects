# -*- coding: utf-8 -*-
"""
Created on Fri May 11 17:22:10 2018

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
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

## Example: Estimating Wine Quality ----
## Step 1 - Data collection : Download the data
wine = pd.read_csv("C:\PritamData\Stats_With_R\Homework\jj.csv")

## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nNo of rows and columns\n", wine.shape)               ## No of rows and columns
print("\nDatatypes \n", wine.dtypes)              ## Structure

X = wine.iloc[:,0:10]
y = wine.iloc[:,11]

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

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
print("Decision Tree Regressor :", regressor)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = regressor.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")

## R-sqaure using classifier function
print("Training R-square : ", regressor.score(X_train, y_train, sample_weight=None))
print("Testing R-square : ", regressor.score(X_test, y_test, sample_weight=None))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# The mean squared error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## Step 5: Model Improvement : Improving model performance ----
regressor = DecisionTreeRegressor(random_state=1, max_depth=5)
regressor.fit(X_train, y_train)
print("\n\n Improved Decision Tree Regressor Model:\n", regressor)
t0=time()
y_pred = regressor.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Training R-square : ", regressor.score(X_train, y_train, sample_weight=None))
print("Testing R-square : ", regressor.score(X_test, y_test, sample_weight=None))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

# The mean squared error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#import graphviz
#dot_data = tree.export_graphviz(regressor, out_file=None,
#                         feature_names=wine.feature_names,
#                         class_names=wine.target_names,
#                         filled=True, rounded=True,
#                         special_characters=True)
#graph = graphviz.Source(dot_data)
#graph.render("wine") # tree saved to wine.pdf