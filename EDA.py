# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:29:30 2018

@author: emota
"""

#EDA - Exaploratory Data Analysis

#%%

import pandas as pd
import numpy as np

iris_filename = "C:\\Users\\emota\\Documents\\DataScience\\9781786462138_Code\\PythonDataScienceEssentialsSecondEdition_Code\\Chapter 2\\code\\datasets-uci-iris.csv"

iris = pd.read_csv(iris_filename, header=None, names= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])

iris.head()

#%%

iris.describe()


#%%
boxes =  iris.boxplot(return_type = 'axes')

#%%

iris.quantile([0.1, 0.9])

#%%

# .median(), .mean(), .std()

iris.target.unique()

iris.sepal_width.median()

#%%

#similarity matrix, be able to see a table of values among different columns

pd.crosstab(iris['petal_length'] > 3.758667, iris['petal_width'] > 1.198667)


#%%

#s - size of the marker
#c - color of the marker
#edgecolor - trim color

scatterplot = iris.plot(kind='scatter', x='petal_width', y='petal_length', s=64, c='blue', edgecolors='white')

#%%

#number of bins = square root of the number of observations

distr = iris.petal_width.plot(kind='hist', alpha=0.5, bins=20)

#%%

