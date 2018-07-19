# -*- coding: utf-8 -*-
"""
Created on Tue May 22 21:22:23 2018

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
from sklearn.neural_network import MLPRegressor

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
### Using MLPRegressor from nueral net
nnet = MLPRegressor(hidden_layer_sizes=(1, ), activation='relu', solver='adam', 
                    alpha=0.0001, batch_size='auto', learning_rate='constant', 
                    learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                    shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                    warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                    early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                    beta_2=0.999, epsilon=1e-08)
nnet.fit(X_train, y_train)
print("\n\nMLPRegressor :", nnet)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = nnet.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")

# Explained variance score: 1 is perfect prediction
print('\nVariance or R-Square score: %.2f' % r2_score(y_test, y_pred))

# The mean squared error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

## Step 5: Model Improvement : Improving model performance ----

### Using MLPRegressor from nueral net
nnet = MLPRegressor(hidden_layer_sizes=(5, ), activation='tanh', solver='adam', 
                    alpha=0.0001, batch_size='auto', learning_rate='constant', 
                    learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                    shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                    warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                    early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                    beta_2=0.999, epsilon=1e-08)
nnet.fit(X_train, y_train)
print("\n\nMLPRegressor :", nnet)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = nnet.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")

# Explained variance score: 1 is perfect prediction
print('\nVariance or R-Square score: %.2f' % r2_score(y_test, y_pred))

# The mean squared error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

### Using MLPRegressor from nueral net
nnet = MLPRegressor(hidden_layer_sizes=(100,100), activation='tanh', solver='adam', 
                    alpha=0.0001, batch_size='auto', learning_rate='constant', 
                    learning_rate_init=0.001, power_t=0.5, max_iter=200, 
                    shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                    warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                    early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                    beta_2=0.999, epsilon=1e-08)
nnet.fit(X_train, y_train)
print("\n\nMLPRegressor using 100, 100 nodes and layers :", nnet)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = nnet.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")

# Explained variance score: 1 is perfect prediction
print('\nVariance or R-Square score: %.2f' % r2_score(y_test, y_pred))

# The mean squared error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
