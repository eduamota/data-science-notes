# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 08:48:48 2018

@author: emota
"""

#%%

import numpy as np

#apply operations element wise

a = np.arange(5).reshape(1,5)
a += 1
a*a

#%%

#boradcasting - repeating dimension into the needed dimensions

a = np.arange(5).reshape(1,5) + 1
b = np.arange(5).reshape(5,1) + 1
			 
a*b

#%%
#same as above
a2 = np.array([1,2,3,4,5]*5).reshape(5,5)

b2 = a2.T

a2*b2

#%%

#add and multiply


np.sum(a2, axis=0) #sum column wise

np.sum(a2, axis=1) #sum row wise

	  
#%%

#On IPython, %time allows you to easily benchmark operations. The -n 1 parameter just requires the benchmark to execute the code snippet for only one loop; -r 3 requires you to retry the execution of the loops (in this case, just one loop) three times and report the best performance recorded from such repetitions.

%timeit -n 1 -r 3 [i+1.0 for i in range(10**6)]
										
%timeit -n 1 -r 3 np.arange(10**6)+1.0
						   
#%%

#Matrix operations

M = np.arange(5*5, dtype=float).reshape(5,5)
M

#%%

coefs = np.array([1., 0.5, 0.5, 0.5, 0.5])
coefs_matrix = np.column_stack((coefs, coefs[::-1]))

print(coefs_matrix)

#%%

np.dot(M, coefs)

#%%
#broadcast the array coefs into a matrix repeats column 1 by the number of columns needed
np.dot(coefs, M)

#%%

np.dot(M, coefs_matrix)

#%%
#slicing and indexing

M = np.arange(100, dtype=int).reshape(10,10)

#slicing as follows
#[start_index_included:end_index_exclude:steps]

M[2:9:2,:]

#%%

#further slice the columns
M[2:9:2, 5:]

#%%

#negatives in the steps, count backwards

M[2:9:2,5::-1]

#%%

#We cannot contextually use Boolean indexes on both columns and rows in the same square brackets, though we can apply the usual indexing to the other dimension using integer indexes

row_index = (M[:,0]>=20) & (M[:,0]<=80)

col_index = M[0,:]>=5
			 
M[row_index,:][:,col_index]

#%%

#If we need a global selection of elements in the array, we can also use a mask of Boolean values, as follows:
	
mask = (M>=20) & (M<=90) & ((M / 10.) % 1 >= 0.5)

M[mask]

#%%

#providing a sequence of integer indexes, where integers may be in a particular order or might even be repeated. Such an approach is called fancy indexing:
	
#elemt-wise opeeration

row_index = [1,1,2,7]
col_index = [0,2,4,8]

M[row_index, col_index]	

#%%

#column and row wise operation

M[row_index,:][:,col_index]


#%%

#slice and index are operations in views, to actually modify the data it needs to be copied

N = M[2:9:2,5:].copy()

#%%

#Stacking Numpy arrays

dataset = np.arange(50).reshape(10,5)
dataset
#%%
single_line = np.arange(1*5).reshape(1,5)
a_few_lines = np.arange(3*5).reshape(3,5)

#vstack adds a row
np.vstack((dataset, single_line))

#%%
np.vstack((dataset, a_few_lines))

#%%

#can add more than one vector
np.vstack((dataset, single_line, single_line, single_line))


#%%

#use hstack to add columns
bias = np.ones(10).reshape(10,1)
np.hstack((dataset,bias))

#%%
#add a column wihtouth the need to reshape it
bias = np.ones(10)
np.column_stack((dataset, bias))

#%%

#dstack will add a depth wise on a 3 dimensional matrix

np.dstack((dataset*1, dataset*2, dataset*3))

#%%

#add a row or column on specific location

np.insert(dataset, 3, bias, axis=1)

#%%

#can insert other matices, but the size needs to match

np.insert(dataset, 3, dataset.T, axis=1)

#%%

np.insert(dataset, 3, np.ones(5), axis=0)