# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 22:31:00 2018

@author: emota
"""

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


#%%

#Load and prepare evaluation of model with and wihtout additonal featuers
cali = datasets.california_housing.fetch_california_housing()

X = cali['data']
Y = cali['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

#%%

#Model wihtout additional features, higher error than with addtional features

from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor()
regressor.fit(X_train, Y_train)
Y_est = regressor.predict(X_test)

print ("MAE=", mean_squared_error(Y_test, Y_est))

#%%

#Model with additional features lower error than with less features

#Do Z-normalization as a feature, this improves accurracy

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

regressor = KNeighborsRegressor()

regressor.fit(X_train_scaled, Y_train)

Y_test = regressor.predict(X_test_scaled)

print ("MAE=", mean_squared_error(Y_test, Y_est))

#%%

#Use Robust Scalar which uses median and IQR

from sklearn.preprocessing import RobustScaler

scaler2 = RobustScaler()
X_train_scaled = scaler2.fit_transform(X_train)
X_test_scaled = scaler2.transform(X_test)

regressor = KNeighborsRegressor()
regressor.fit(X_train_scaled, Y_train)

Y_est = regressor.predict(X_test_scaled)

print ("MAE=", mean_squared_error(Y_test, Y_est))

#%%

non_linear_feat = 5 # AveOccup column

X_train_new_feat = np.sqrt(X_train[:, non_linear_feat]) #new feat is the square root of the average occupancy column

X_train_new_feat.shape = (X_train_new_feat.shape[0], 1)

X_train_extended = np.hstack([X_train, X_train_new_feat])

X_test_new_feat = np.sqrt(X_test[:, non_linear_feat])

X_test_new_feat.shape = (X_test_new_feat.shape[0], 1)

X_test_extended = np.hstack([X_test, X_test_new_feat])

scaler = StandardScaler()

X_train_extended_scaled = scaler.fit_transform(X_train_extended)

X_test_extended_scaled = scaler.transform(X_test_extended)

regressor = KNeighborsRegressor()

regressor.fit(X_train_extended_scaled, Y_train)

Y_est = regressor.predict(X_test_extended_scaled)


print ("MAE=", mean_squared_error(Y_test, Y_est)) 

#how important it is for your analysis the application of linear and nonlinear transformations found by EDA and obtain features that are conceptually more related to the output variable.