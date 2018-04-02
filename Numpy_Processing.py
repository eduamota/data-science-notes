# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:04:27 2018

@author: emota
"""

#%%
import numpy as np

'''
Advantages:
	
It is memory optimal (and, besides other aspects, configured to transmit data to C or Fortran routines in the best-performing layout of memory blocks)

It allows fast linear algebra computations (vectorization) and element-wise operations (broadcasting) without any need to use iterations with for loops, which is generally computationally expensive in Python

Critical libraries, such as SciPy or scikit-learn, expect arrays as an input for their functions to operate correctly

Limitations:
	
	They usually store only elements of a single, specific data type, which you can define beforehand (but there's a way to define complex data and heterogeneous data types, though they could be very difficult to handle for analysis purposes).

After they are initialized, their size is fixed. If you want to change their shape, you have to create them anew.

Ways to create a numpy array:
	
Transforming an existing data structure into an array

Creating an array from scratch and populating it with default or calculated values

Uploading some data from a disk into an array

'''
#%%

#Transform a list into a uni-dimensional array

list_of_ints = [1,2,3]

Array_1 = np.array(list_of_ints)

Array_1

#%%

type(Array_1)

Array_1.dtype

#%%

#evalaute the size in memory
Array_1.nbytes

#%%

Array_1 = np.array(list_of_ints, dtype = 'int8')

Array_1.nbytes

'''
Type

Size in bytes

Description

bool

1

Boolean (True or False) stored as a byte

int_

4

Default integer type (normally int32 or int64)

int8

1

Byte (-128 to 127)

int16

2

Integer (-32,768 to 32,767)

int32

4

Integer (-2**31 to 2**31-1)

int64

8

Integer (-2**63 to 2**63-1)

uint8

1

Unsigned integer (0 to 255)

uint16

2

Unsigned integer (0 to 65,535)

uint32

3

Unsigned integer (0 to 2**32-1)

uint64

4

Unsigned integer (0 to 2**64-1)

float_

8

Shorthand for float64

float16

2

Half-precision float (exponent 5 bits, mantissa 10 bits)

float32

4

Single-precision float (exponent 8 bits, mantissa 23 bits)

float64

8

Double-precision float (exponent 11 bits, mantissa 52 bits)

'''
#%%

Array_1b = Array_1.astype('float32')

Array_1b

#%%

# Transform a list into a bidimensional array
a_list_of_lists = [[1,2,3],[4,5,6],[7,8,9]]
Array_2D = np.array(a_list_of_lists )
Array_2D

#%%

# Transform a list into a multi-dimensional array
a_list_of_lists_of_lists = [[[1,2],[3,4],[5,6]],
[[7,8],[9,10],[11,12]]]
Array_3D = np.array(a_list_of_lists_of_lists)
Array_3D

#%%

#convert dict to numpy array

np.array({1:2,3:4,5:6}.items())

#%%

# Restructuring a NumPy array shape
original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
Array_a = original_array.reshape(4,2)
Array_b = original_array.reshape(4,2).copy()
Array_c = original_array.reshape(2,2,2)
# Attention because reshape creates just views, not copies
original_array[0] = -1
			  
#%%

ordinal_values = np.arange(9).reshape(3,3)

np.arange(9)[::-1]

np.random.randint(low=1,high=10,size=(3,3)).reshape(3,3)

np.zeros((3,3))

np.ones((3,3))

np.eye(3)

fractions = np.linspace(start=0, stop=1, num=10)

growth = np.logspace(start=0, stop=1, num=10, base=10.0)

#%%

std_gaussian = np.random.normal(size=(3,3))

gaussian = np.random.normal(loc=1.0, scale= 3.0, size=(3,3))

rand = np.random.uniform(low=0.0, high=1.0, size=(3,3))

#%%

housing = np.loadtxt('C:\\Users\\emota\\Documents\\DataScience\\9781786462138_Code\\PythonDataScienceEssentialsSecondEdition_Code\\Chapter 2\\code\\regression-datasets-housing.csv',delimiter=',', dtype=float)


#%%
import pandas as pd

housing = pd.read_csv('C:\\Users\\emota\\Documents\\DataScience\\9781786462138_Code\\PythonDataScienceEssentialsSecondEdition_Code\\Chapter 2\\code\\regression-datasets-housing.csv', header=None)

housing_array = housing.values

housing_array.dtype