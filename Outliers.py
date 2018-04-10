# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:11:36 2018

@author: emota
"""

#%%

import pandas as pd
import numpy as np

#%%

#The detection and treatment of outliers

#Univariate outlier detection, looking at one variable at a time

'''
Outliers are:
	*If you are observing Z-scores, observations with scores higher than 3 in absolute value have to be considered as suspect outliers.

	*If you are observing a description of data, you can consider as suspect outliers the observations that are smaller than the 25th percentile value minus the IQR (the interquartile range, that is, the difference between the 75th and 25th percentile values) *1.5 and those greater that the 75th percentile value plus the IQR * 1.5. Usually, you can achieve such distinction with the help of a boxplot graph.
'''

from sklearn.datasets import load_boston

boston = load_boston()

continuous_variables = [n for n in range(boston.data.shape[1]) if n != 3]

#use StandardScaler to normalize the data tozero mean and unit variance

from sklearn import preprocessing

normalized_data = preprocessing.StandardScaler().fit_transform(boston.data[:, continuous_variables])

outliers_rows, outliers_columns = np.where(np.abs(normalized_data)>3)

print (list(zip(outliers_rows, outliers_columns)))

#%%

#For multivariate use algorithms to reduce dimensionality and then analyze outliers

'''
Use the following algorithms:
	The covariance.EllipticEnvelope class fits a robust distribution estimation of your data, pointing out the outliers that might be contaminating your dataset because they are the extreme points in the general distribution of the data.

The svm.OneClassSVM class is a support vector machine algorithm that can approximate the shape of your data and find out if any new instances provided should be considered as a novelty (it acts as a novelty detector because by default, it presumes that there is no outlier in the data). Anyway, by just modifying its parameters, it can also work on a dataset where outliers are present, providing an even more robust and reliable outlier detection system than EllipticEnvelope.
'''

#%%

#EllipticEnvelope Evaluates every point to a great mean from all variables so it works for univariate and multivariate

#contamination value upto 0.5, start with 0.01 to 0.02

#Create an artificial distribution made of blobs

from sklearn.datasets import make_blobs

blobs = 1

blob = make_blobs(n_samples=100, n_features=2, centers=blobs, cluster_std=1.5, shuffle=True, random_state=5)

#Robust Covariance Estimates

from sklearn.covariance import EllipticEnvelope

robust_covariance_est = EllipticEnvelope(contamination=0.1).fit(blob[0])

detection = robust_covariance_est.predict(blob[0])

outliers = np.where(detection==-1)[0]

inliers = np.where(detection==1)[0]

#Draw the distribution and the detected outliers
from matplotlib import pyplot as plt

#Just the distribution

plt.scatter(blob[0][:,0], blob[0][:,1], c='blue', alpha=0.8, s=60, marker='o', edgecolors='white')

plt.show()

#The distribution and the outliers

in_points = plt.scatter(blob[0][inliers,0], blob[0][inliers,1], c='blue', alpha=0.8, s=60, marker='o', edgecolors='white')

out_points = plt.scatter(blob[0][outliers, 0], blob[0][outliers, 1], c='red', alpha=0.8, s=60, marker='o', edgecolors='white')

plt.legend((in_points, out_points), ('inliers', 'outliers'), scatterpoints=1, loc='lower right')

plt.show()