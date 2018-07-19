# -*- coding: utf-8 -*-
"""
Created on Sat May 19 23:37:06 2018

@author: prita
"""

import sys
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

## Example: Space Shuttle Launch Data ----
## Step 1 - Data collection : Download the data
launch = pd.read_csv("C:\PritamData\Stats_With_R\Homework\challenger.csv")

## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nNo of rows and columns\n", launch.shape)               ## No of rows and columns
print("\nDatatypes \n", launch.dtypes)              ## Structure

X = launch.iloc[:,1:4]
y = launch.iloc[:,4]

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("\nSize of Training Dataset\n",X_train.shape)     ## count for X_train
print("\nSize of Testing Dataset\n",X_test.shape)     ## count for X_test

print("\nSize of Training Dataset\n",y_train.count())       ## count for y_train
print("\nSize of Testing Dataset\n",y_test.count()) 

## Step 3: Model Training:  Training a model on the data ----
logitClassifier = LogisticRegression()
logitClassifier.fit(X_train, y_train)
print("Logistic Regression Classifier :", logitClassifier)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = logitClassifier.predict(X_test)
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
print(logitClassifier.score(X_train, y_train, sample_weight=None))
print("\nTesting Accuracy : ")
print(logitClassifier.score(X_test, y_test, sample_weight=None))      


## Example: Logistic Regression Using Credit Data ----
## Step 1 - Data collection : Download the data
credit = pd.read_csv("C:\PritamData\Stats_With_R\Homework\credit.csv")

## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nNo of rows and columns\n", credit.shape)               ## No of rows and columns
print("\nDatatypes \n", credit.dtypes)              ## Structure

X = credit.iloc[:,0:16]
y = credit.iloc[:,16]

## Feature Scaling
# Encoding categorical data or Independent Variable
labelencoder_x = LabelEncoder()
X.checking_balance = labelencoder_x.fit_transform(X.checking_balance)
X.credit_history = labelencoder_x.fit_transform(X.credit_history)
X.purpose = labelencoder_x.fit_transform(X.purpose)
X.savings_balance = labelencoder_x.fit_transform(X.savings_balance)
X.employment_duration = labelencoder_x.fit_transform(X.employment_duration)
X.other_credit = labelencoder_x.fit_transform(X.other_credit)
X.housing = labelencoder_x.fit_transform(X.housing)
X.job = labelencoder_x.fit_transform(X.job)
X.phone = labelencoder_x.fit_transform(X.phone)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)        ### No = 0, Yes = 1

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("\nSize of Training Dataset\n",X_train.shape)     ## count for X_train
print("\nSize of Testing Dataset\n",X_test.shape)     ## count for X_test

print("\nSize of Training Dataset\n",len(y_train))       ## count for y_train
print("\nSize of Testing Dataset\n",len(y_test)) 

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('CART', DecisionTreeClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("\n[%s] Mean: %.8f Std. Dev.: %8f" %(name, cv_results.mean(), cv_results.std()))

## Step 3: Model Training:  Training a model on the data ----
logitClassifier = LogisticRegression()
logitClassifier.fit(X_train, y_train)
print("Logistic Regression Classifier :", logitClassifier)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = logitClassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("\nConfusion matrix after prediction")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nAccuracy %: ")
print(accuracy_score(y_test, y_pred)*100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))                                                  #(tn, fp, fn, tp)

## Accuracy using classifier function
print("\nTraining Accuracy : ")
print(logitClassifier.score(X_train, y_train, sample_weight=None))
print("\nTesting Accuracy : ")
print(logitClassifier.score(X_test, y_test, sample_weight=None))      

print ("Training R_square = ", logitClassifier.score(X_train,y_train))
print ("Testing R_square = ", logitClassifier.score(X_test, y_test))
#print('Coefficients or Slopes: \n', reg2.coef_)
print ("Intercept = ", logitClassifier.intercept_)
#coeff_df = pd.DataFrame(logitClassifier.coef_ , X_train.columns)  
#print("Coefficients", coeff_df)
print("Coefficients", logitClassifier.coef_)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

false_positive_rate, recall, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()

## Step 5: Model Improvement : Improving model performance using GridSearch ----

pipeline = Pipeline([
            ('clf', LogisticRegression())
            ])
parameters = {
            'C': (0.1, 1, 10)
            }
grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='recall')
grid_search.fit(X_train, y_train)
print('Best score: %0.3f' % grid_search.best_score_)
print ('Best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
