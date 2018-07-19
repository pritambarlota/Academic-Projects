# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:20:10 2018

@author: prita
"""
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

## Step 1 - Data collection : Download the data
credit = pd.read_csv("C:\PritamData\Stats_With_R\Homework\credit.csv")

## Step 2: Data exploration and preparation : Exploring and preparing the data ---- 
print("\nNo of rows and columns\n", credit.shape)               ## No of rows and columns
print("\nDatatypes \n", credit.dtypes)              ## Structure
print("\nCount of default\n",credit.groupby('default').size())       ## count
print("\nLenght ", len(credit.default))
print("\nCount of Checking Balance\n",credit.groupby('checking_balance').size()) 
print("\nCount of Savings Balance\n",credit.groupby('savings_balance').size())    
print("\nSummary of Numeric Variables\n",credit.describe(include=[np.number]))    
     
X = credit.iloc[:,:16]
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

## BootStrapping
## Step 3: Model Training:  Training a model on the data ----
### Using RandomForestClassifier and criterion='entropy'
ranforestclassifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=4, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=5, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
ranforestclassifier.fit(X_train, y_train)
print("\n\nRandom Forest Classifier :", ranforestclassifier)

## Step 4: Model Evaluation : Evaluating model performance ----
t0=time()
y_pred = ranforestclassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: \n",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", ranforestclassifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", ranforestclassifier.score(X_test, y_test, sample_weight=None))
seed=7
scoring = 'recall'
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(ranforestclassifier, X_train, y_train, cv=kfold, scoring=scoring)
print("10 Fold Scores are : ", cv_results)                             
print("mean score is :", cv_results.mean())  

#Random forest to compute the feature importances

importances = ranforestclassifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in ranforestclassifier.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

extratreesclassifier = ExtraTreesClassifier(n_estimators=10, criterion='entropy', 
        max_depth=4, min_samples_split=2, min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, max_features='auto', 
        max_leaf_nodes=None, min_impurity_decrease=0.0, 
        min_impurity_split=None, bootstrap=False, oob_score=False, 
        n_jobs=1, random_state=None, verbose=0, warm_start=False, 
        class_weight=None)
extratreesclassifier.fit(X_train, y_train)
print("\n\nExtra Trees Classifier :", extratreesclassifier)
t0=time()
y_pred = extratreesclassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: \n",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", extratreesclassifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", extratreesclassifier.score(X_test, y_test, sample_weight=None))
cv_results = cross_val_score(extratreesclassifier, X_train, y_train, cv=kfold, scoring=scoring)
print("10 Fold Scores are : ", cv_results)                             
print("mean score is :", cv_results.mean())             

## Boosting

adaBoostclassifier = AdaBoostClassifier(n_estimators=10, learning_rate=1.0, random_state=0)
adaBoostclassifier.fit(X_train, y_train)
print("\n\nAdaBoost Classifier :", adaBoostclassifier)
t0=time()
y_pred = adaBoostclassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: \n",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", adaBoostclassifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", adaBoostclassifier.score(X_test, y_test, sample_weight=None))
cv_results = cross_val_score(adaBoostclassifier, X_train, y_train, cv=kfold, scoring=scoring)
print("10 Fold Scores are : ", cv_results)                             
print("mean score is :", cv_results.mean())     


gradientboostingclassifier = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,
    max_depth=4, random_state=0)
gradientboostingclassifier.fit(X_train, y_train)
print("\n\nGradient Boosting Classifier :", gradientboostingclassifier)
t0=time()
y_pred = gradientboostingclassifier.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: \n",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", gradientboostingclassifier.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", gradientboostingclassifier.score(X_test, y_test, sample_weight=None))
cv_results = cross_val_score(gradientboostingclassifier, X_train, y_train, cv=kfold, scoring=scoring)
print("10 Fold Scores are : ", cv_results)                             
print("mean recall score is :", cv_results.mean())             

## Bagging
bagger = BaggingClassifier(n_estimators=10, random_state=0)
bagger.fit(X_train, y_train)
print("\n\nBagging Classifier :", bagger)
t0=time()
y_pred = bagger.predict(X_test)
print("\nPredictions time:", round(time()-t0, 3), "s")
print("Confusion matrix after prediction\n", confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report: ",classification_report(y_test, y_pred))
## Accuracy using classifier function
print("Training Accuracy : ", bagger.score(X_train, y_train, sample_weight=None))
print("Testing Accuracy : ", bagger.score(X_test, y_test, sample_weight=None))
cv_results = cross_val_score(bagger, X_train, y_train, cv=kfold, scoring=scoring)
print("10 Fold Scores are : ", cv_results)                             
print("mean recall score is :", cv_results.mean())  





#import graphviz 
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("iris") 
