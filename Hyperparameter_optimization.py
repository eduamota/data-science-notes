# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 22:21:24 2018

@author: emota
"""
#%%
import pandas as pd
import numpy as np

#%%

#load the handwritten data

from sklearn.datasets import load_digits
digits = load_digits()

X, y = digits.data, digits.target

#%%

from sklearn import svm

h = svm.SVC()
hp = svm.SVC(probability=True, random_state=1)

#%%

from sklearn.model_selection import GridSearchCV

search_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel':['rbf']},]

scorer = 'accuracy'

#parameters can be created dynamically {'C' :np.logspace(start=-2, stop=3, num=6, base=10.0)}

#%%

search_func = GridSearchCV(estimator=h, param_grid=search_grid, scoring=scorer, n_jobs=-1, iid=False, refit=True, cv=10)

%timeit search_func.fit(X,y)

print (search_func.best_estimator_)
print (search_func.best_params_)
print (search_func.best_score_)

#%%

from sklearn.metrics import log_loss, make_scorer
Log_Loss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

#%%

search_func = grid_search.GridSearchCV(estimator=hp, param_grid=search_grid, scoring=Log_Loss, n_jobs=-1, iid=False, refit=True, cv=3)
search_func.fit(X,y)
print (search_func.best_score_)
print (search_func.best_params_)

#%%

import numpy as np

from sklearn.preprocessing import LabelBinarizer

def my_custom_log_loss_func(ground_truth, p_predictions, penalty = list(), eps=1e-15):
	adj_p = np.clip(p_predictions, eps, 1 - eps)
	lb = LabelBinarizer()
	g = lb.fit_transform(ground_truth)
	if g.shape[1] == 1:
		g = np.append(1 - g, g, axis=1)
	if penalty:
		g[:,penalty] = g[:,penalty] * 2
	summation = np.sum(g * np.log(adj_p))
	return summation * (-1.0/len(ground_truth))

#%%

my_custom_scorer = make_scorer(my_custom_log_loss_func, greater_is_better=False, needs_proba=True, penalty = [4,9])

#%%

from sklearn import grid_search
search_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
search_func = grid_search.GridSearchCV(estimator=hp, param_grid=search_grid, scoring=my_custom_scorer, n_jobs=1, iid=False, cv=3)
search_func.fit(X,y)
print (search_func.best_score_)
print (search_func.best_params_)

#%%
search_dict = {'kernel': ['linear','rbf'],'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
scorer = 'accuracy'
search_func = grid_search.RandomizedSearchCV(estimator=h, param_distributions=search_dict, n_iter=7, scoring=scorer, n_jobs=-1, iid=False, refit=True, cv=10)
%timeit search_func.fit(X,y)
print (search_func.best_estimator_)
print (search_func.best_params_)
print (search_func.best_score_)

#%%

search_func.grid_scores_
