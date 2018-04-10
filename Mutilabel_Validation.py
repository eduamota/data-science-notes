# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 22:21:26 2018

@author: emota
"""

#%%

import pandas as pd
import numpy as np

#%%

from sklearn import datasets

iris = datasets.load_iris()

#No crossvalidation for this dummy notebook

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.50, random_state=4)

#Use a bad multiclass classifier
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=2)

classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

iris.target_names

#%%

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

print(cm)

#%%

import matplotlib.pyplot as plt

img = plt.matshow(cm, cmap=plt.cm.autumn)

plt.colorbar(img, fraction=0.045)

for x in range(cm.shape[0]):
	for y in range(cm.shape[1]):
		plt.text(x, y, "%0.2f" % cm[x,y], size=12, color='black', ha='center', va='center')

plt.show()


#%%

from sklearn import metrics

print ('Accuracy:', metrics.accuracy_score(Y_test, Y_pred))

#%%

print ("Precision:", metrics.precision_score(Y_test, Y_pred, average='micro'))

#%%

print ("Recall:", metrics.recall_score(Y_test, Y_pred, average='micro'))

#%%

print ("F1 score:", metrics.f1_score(Y_test, Y_pred, average='micro'))

#%%

from sklearn.metrics import classification_report

print (classification_report(Y_test, Y_pred, target_names=iris.target_names))

#%%

'''In data science practice, Precision and Recall are used more extensively than Accuracy as most datasets in data problems tend to be unbalanced. To account for this imbalance, data scientists often present their results in terms of Precision, Recall, and F1-score.'''

#%%

#Binary classification

#Receiver Operating Characteristics curve


#Function to computer AUC sklearn.metrics.roc_auc_score()

#%%
#Regression

from sklearn.metrics import mean_absolute_error

mean_absolute_error([1.0, 0.0, 0.0], [0.0, 0.0, -1.0])

#%%

from sklearn.metrics import mean_squared_error

mean_squared_error([-10.0, 0.0, 0.0], [0.0, 0.0, 0.0])

#%%

#R2 determines how good a linear fit is that exists between the predictors and the target variable. It takes values between 0 and 1 (inclusive); the higher R2 is, the better the model.

#sklearn.metrics.r2_score
