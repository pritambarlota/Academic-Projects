# -*- coding: utf-8 -*-
"""
Created on Tue May 22 20:08:46 2018

@author: prita
"""

import sys
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

## Step 1 - Data collection : Download the data
concrete = pd.read_csv("C:\PritamData\Stats_With_R\Homework\concrete.csv")

## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nNo of rows and columns\n", concrete.shape)               ## No of rows and columns
print("\nDatatypes \n", concrete.dtypes)              ## Structure
print("\nLenght ", len(concrete.strength))
print("\nSummary of Numeric Variables\n",concrete.describe(include=[np.number]))    

plt.hist(concrete.iloc[:,8], alpha=0.5, bins=50 )
plt.xlabel("strength")

## Feature Scaling
sc = StandardScaler()
concrete = pd.DataFrame(sc.fit_transform(concrete))
print("Data after Standardization")
print(concrete.head(5))    

## split dataset into X and y
X = concrete.iloc[:,:8]
y = concrete.iloc[:,8]

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("\nSize of Training Dataset\n",X_train.shape)     ## count for X_train
print("\nSize of Testing Dataset\n",X_test.shape)     ## count for X_test

print("\nSize of Training Dataset\n",len(y_train))       ## count for y_train
print("\nSize of Testing Dataset\n",len(y_test)) 

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

#cov_concrete = pd.DataFrame(y_test, y_pred)
#print("Covariance between actual and predicted :", cov_concrete.corr())



## Step 5: Model Improvement : Evaluating model performance ----
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
