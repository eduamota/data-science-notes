# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:58:43 2018

@author: emota
"""

#%%
import pandas as pd


#%%
iris_filename = "C:\\Users\\emota\\Documents\\DataScience\\9781786462138_Code\\PythonDataScienceEssentialsSecondEdition_Code\\Chapter 2\\code\\datasets-uci-iris.csv"

iris = pd.read_csv(iris_filename, sep=',', decimal='.', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])

#%%

iris.head()

#%%
iris.tail()

#%%

iris.head(2)

#%%

iris.columns
#the result is a pandas index not a list

#%%
Y = iris['target']
Y

#%%

X = iris[['sepal_length', 'sepal_width']]

#%%
print(X.shape)
print(Y.shape)

#%%

fake_dataset = pd.read_csv("C:\\Users\\emota\\Documents\\DataScience\\9781786462138_Code\\PythonDataScienceEssentialsSecondEdition_Code\\Chapter 2\\code\\a_loading_example_1.csv", sep=',')

fake_dataset

#%%

fake_dataset = pd.read_csv("C:\\Users\\emota\\Documents\\DataScience\\9781786462138_Code\\PythonDataScienceEssentialsSecondEdition_Code\\Chapter 2\\code\\a_loading_example_1.csv", sep=',', parse_dates=[0])

fake_dataset

#%%

#Note that this method only fills missing values in the view of the data (that is, it doesn't modify the original DataFrame). In order to actually change them, use the inplace=True argument.
fake_dataset.fillna(50)

#%%

#because the change happened on the view, this will run properly even after the data was modified on the previous line.
fake_dataset.fillna(-1)

#%%

#Note that axis=0 implies a calculation of means that spans the rows; the consequently obtained means are derived from column-wise computations. Instead, axis=1 spans columns and, therefore, row-wise results are obtained. This works in the same way for all other methods that require the axis parameter, both in pandas and NumPy.
fake_dataset.fillna(fake_dataset.mean(axis=0))

#%%

#Ignore bad lines, onlye when there is not a lot

bad_dataset = pd.read_csv("C:\\Users\\emota\\Documents\\DataScience\\9781786462138_Code\\PythonDataScienceEssentialsSecondEdition_Code\\Chapter 2\\code\\a_loading_example_2.csv", error_bad_lines=False)

bad_dataset

#%%

#Dealing with big datasets

#Chunks

iris_chunks = pd.read_csv(iris_filename, header=None, names=['C1', 'C2', 'C3', 'C4', 'C5'], chunksize=10)

for chunk in iris_chunks:
	print('Shape:', chunk.shape)
	print(chunk, '\n')
	
#%%

#Ask for an iterator

iris_iterator = pd.read_csv(iris_filename, header=None, names=['C1', 'C2', 'C3', 'C4', 'C5'], iterator=True)

print (iris_iterator.get_chunk(10).shape)

print (iris_iterator.get_chunk(20).shape)

piece = iris_iterator.get_chunk(2)
piece

#%%

import numpy as np
import csv

def batch_read(filename, batch=5):
	# open the data stream
	with open(filename, 'rt') as data_stream:
	# reset the batch
		batch_output = list()
	# iterate over the file
		for n, row in enumerate(csv.reader(data_stream,
	dialect='excel')):
			# if the batch is of the right size
			if n > 0 and n % batch == 0:
				# yield back the batch as an ndarray
				yield(np.array(batch_output))
				# reset the batch and restart
				batch_output = list()
				# otherwise add the row to the batch
				batch_output.append(row)
		# when the loop is over, yield what's left
		yield(np.array(batch_output))


#%%
#pandas DataFrames can be created by merging series or other list-like data. Note that scalars are transformed into lists

#this process does not accept list of multiple sizes

my_own_dataset = pd.DataFrame({'Col1': range(5), 'Col2':
[1.0]*5, 'Col3': 1.0, 'Col4': 'Hello World!'})
my_own_dataset

#%%

#chek data types

my_own_dataset.dtypes

#%%

#Cast columns from one type to antoher

my_own_dataset['Col1'] = my_own_dataset['Col1'].astype(float)

my_own_dataset.dtypes

#%%

#if you need to apply a function to a limited section of rows, you can create a mask. A mask is a series of Boolean values (that is, True or False) that tells whether the line is selected or not.

mask_feature = iris['sepal_length'] > 6.0
mask_feature

#%%

mask_target = iris['target'] == 'Iris-virginica'
iris.loc[mask_target, 'target'] = 'New label'

#%%

iris['target'].unique()

#%%

grouped_targets_mean = iris.groupby(['target']).mean()

grouped_targets_mean

#%%

grouped_targets_var = iris.groupby(['target']).var()
grouped_targets_var

#%%

iris.sort_index(by='sepal_length').head()

#%%
#if your dataset contains a time series (for example, in the case of a numerical target) and you need to apply a rolling operation to it (in the case of noisy data points)

smooth_time_series = pd.rolling_mean(time_series, 5)

#%%
#the apply() pandas method is able to perform any row-wise or column-wise operation programmatically. apply() should be called directly on the DataFrame; the first argument is the function to be applied row-wise or column-wise; the second the axis to apply it on. Note that the function can be a built-in, library-provided, lambda or any other user-defined function.

import numpy as np
iris.apply(np.count_nonzero, axis=1).head()

#%%

#per column
iris.apply(np.count_nonzero, axis=0)



#%%

#to operate element-wise, the applymap() method should be used on the DataFrame. In this case, just one argument should be provided: the function to apply.


iris.applymap(lambda el:len(str(el))).head()

#%%
iris.head()

#%%

dataset = pd.read_csv("C:\\Users\\emota\\Documents\\DataScience\\9781786462138_Code\\PythonDataScienceEssentialsSecondEdition_Code\\Chapter 2\\code\\a_selection_example_1.csv", index_col=0)

dataset

#%%

dataset[['val3', 'val2']][0:2]

dataset.loc[range(100, 102), ['val3', 'val2']]

dataset.ix[range(100, 102), ['val3', 'val2']]

dataset.ix[range(100, 102), [2,1]]

dataset.iloc[range(2) [2,1]]