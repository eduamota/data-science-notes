# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:17:12 2018

@author: emota
"""

#%%

import pandas as pd
import numpy as np

#%%

from sklearn.model_selection import train_test_split

choosen_random_state = 1
cv_folds = 10 # Try 3, 5, or 20
eval_scoring='accuracy' #Try also f1

workers = -1 #this will use all your CPU power

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=choosen_random_state)