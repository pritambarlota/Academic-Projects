# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:59:11 2018

@author: prita
"""

import sys
import numpy as np
import pandas as pd
from time import time
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
dataframe = pd.read_csv("C:\PritamData\Stats_With_R\Homework\iris.csv")
dataframe = dataframe.iloc[:,1:6]

print("\nNo of rows and columns\n", dataframe.shape)               ## No of rows and columns
print("\nDatatypes \n", dataframe.dtypes)
print("\nSummary of Numeric Variables\n",dataframe.describe(include=[np.number]))    

X = dataframe.iloc[:,0:4]
Y = dataframe.iloc[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, 
                             verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy of Model: %.2f%% (%.2f%%)",(results.mean()*100, results.std()*100))