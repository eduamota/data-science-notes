# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 22:34:35 2018

@author: emota
"""

#Testing and validating

import pandas as pd
import numpy as np

#%%

from sklearn.datasets import load_digits

digits = load_digits()

print(digits.DESCR)

X = digits.data
y = digits.target

#%%
X[0]

#%%

from sklearn import svm

h1 = svm.LinearSVC(C=1.0)
h2 = svm.SVC(kernel='rbf', degree=3, gamma=0.001, C=1.0) #Radial basis SVC

h3 = svm.SVC(kernel="poly", degree=3, C=1.0) # 3rd degree polynomial SVC

#%%

h1.fit(X, y)

print(h1.score(X,y))

#%%

#Test 30% size and 70% training

from sklearn.model_selection import train_test_split
chosen_random_state = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=chosen_random_state)

print ("X train shape %s, X test shape %s, \ny train shape %s, y test shape %s"  % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))

h1.fit(X_train, y_train)

print (h1.score(X_test, y_test))
#Returns the mean accuracy on the given test data and labels

#%%


chosen_random_state = 1

X_train, X_validation_test, y_train, y_validation_test = train_test_split(X, y, test_size=.40, random_state=chosen_random_state)

X_validation, X_test, y_validation, y_test = train_test_split(X_validation_test, y_validation_test, test_size=.50, random_state=chosen_random_state)

print ("X train shape, %s, X validation shape %s, X test shape %s, \ny train shape %s, y validation shape %s, y test shape %s\n" % (X_train.shape, X_validation.shape, X_test.shape, y_train.shape, y_validation.shape, y_test.shape))


for hypothesis in [h1, h2, h3]:
	hypothesis.fit(X_train, y_train)

	print ("%s -> validation mean accuracy = %0.3f" % (hypothesis, hypothesis.score(X_validation,y_validation)))

h2.fit(X_train, y_train)

print ("\n%s -> test mean accuracy = %0.3f" % (h2, h2.score(X_test,y_test)))

#%%

#Cross-validation
from sklearn.model_selection import cross_val_score

choosen_random_state = 1
cv_folds = 10 # Try 3, 5 or 20
eval_scoring='accuracy' # Try also f1
workers = 1 # this will use all your CPU power
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=choosen_random_state)

for hypothesis in [h1, h2, h3]:
	scores = cross_val_score(hypothesis, X_train, y_train, cv=cv_folds, scoring=eval_scoring, n_jobs=workers)

	print ("%s -> cross validation accuracy: mean = %0.3f std = %0.3f" % (hypothesis, np.mean(scores), np.std(scores)))

	#%%

	import random
def Bootstrap(n, n_iter=3, random_state=None):
    """
    Random sampling with replacement cross-validation generator.
    For each iter a sample bootstrap of the indexes [0, n) is
    generated and the function returns the obtained sample
    and a list of all the excluded indexes.
    """
    if random_state:
        random.seed(random_state)
    for j in range(n_iter):
        bs = [random.randint(0, n-1) for i in range(n)]
        out_bs = list({i for i in range(n)} - set(bs))
        yield bs, out_bs
boot = Bootstrap(n=100, n_iter=10, random_state=1)
for train_idx, validation_idx in boot:
    print (train_idx, validation_idx)