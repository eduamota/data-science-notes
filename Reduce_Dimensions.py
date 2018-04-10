# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:18:42 2018

@author: emota
"""

import pandas as pd
import numpy as np

#%%

#Dimensionality reduction

'''
The main hypothesis behind many algorithms used in the reduction is the one pertaining to Additive White Gaussian Noise (AWGN) noise. We suppose an independent Gaussian-shaped noise has been added to every feature of the dataset. Consequently, reducing the dimensionality also reduces the energy of the noise since you're decreasing its span set.
'''

#Covariance matrix

'''
t's usually the first step in dimensionality reduction because it gives you an idea of the number of features that are strongly related (and therefore, the number of features that you can discard) and the ones that are independent.
'''

from sklearn import datasets

iris = datasets.load_iris()

cov_data = np.corrcoef(iris.data.T)

print (iris.feature_names)

print(cov_data)

#%%
#Use heatmpa to visualize the covariance matrix

import matplotlib.pyplot as plt

img = plt.matshow(cov_data, cmap=plt.cm.rainbow)

plt.colorbar(img, ticks=[-1, 0, 1], fraction=0.045)

for x in range(cov_data.shape[0]):
	for y in range(cov_data.shape[1]):
		plt.text(x, y, "%0.2f" % cov_data[x,y], size=12, color='black', ha='center', va='center')
		
plt.show()

#%%

#Principal component Analysys (PCA)

#Based on the previous data visualization it was identified that 2 dimensions are the best.

from sklearn.decomposition import PCA

pca_2c = PCA(n_components=2)#number of dimension
X_pca_2c = pca_2c.fit_transform(iris.data)
X_pca_2c.shape

plt.scatter(X_pca_2c[:,0], X_pca_2c[:,1], c=iris.target, alpha=0.8, s=60, marker='o', edgecolors='white')

plt.show()

pca_2c.explained_variance_ratio_.sum()

#%%

#Visualize the PCA-restructure array

pca_2c.components_

#%%

#Using eigenvectors to further increase efficiency

pca_2cw = PCA(n_components=2, whiten=True)

X_pca_1cw = pca_2cw.fit_transform(iris.data)

plt.scatter(X_pca_1cw[:,0], X_pca_1cw[:,1], c=iris.target, alpha=0.8, s=60, marker='o', edgecolors='white')

plt.show()

pca_2cw.explained_variance_ratio_.sum()

#%%

#Using 1 dimension, lower acurracy

pca_1c = PCA(n_components=1)

X_pca_1c = pca_1c.fit_transform(iris.data)
plt.scatter(X_pca_1c[:,0], np.zeros(X_pca_1c.shape), c=iris.target, alpha=0.8, s=60, marker='o', edgecolors='white')

plt.show()

pca_1c.explained_variance_ratio_.sum()

#%%

#Tip to get a dataset with at least 95% of energy or useful features, in the case of the iris data set would generate 2 dimensions as previously analyzed

pca_95pc = PCA(n_components=0.95)

X_pca_95pc = pca_95pc.fit_transform(iris.data)

print (pca_95pc.explained_variance_ratio_.sum())

print (X_pca_95pc.shape)

#%%

#PCA is complex and difficult to scale du to the underlaying Singular Value Decomposition (SVD). A simulation and lighter model is available, Randomized SVD. Randomized SVD is better for larger datasets
									
from sklearn.decomposition import RandomizedPCA

rpca_2c = RandomizedPCA(n_components=2)

X_rpca_2c = rpca_2c.fit_transform(iris.data)

plt.scatter(X_rpca_2c[:,0], X_rpca_2c[:,1], c=iris.target, alpha=0.8, s=60, marker='o', edgecolors='white')

plt.show()

rpca_2c.explained_variance_ratio_.sum()											  
#%%

#Latent Factor Analysis

#Similar to PCA but with no orthogonal decomposition, and can be consider a generalization of the PCA algorithm. Uses an Arbitrary Waveform Generator (AWG)

#Good for large datasets

from sklearn.decomposition import FactorAnalysis

fact_2c = FactorAnalysis(n_components=2)

X_factor = fact_2c.fit_transform(iris.data)

plt.scatter(X_factor[:,0], X_factor[:,1], c=iris.target, alpha=0.8, s=60, marker='o', edgecolors='white')

plt.show()

#%%

#Linear Dsicriminant Analysys (LDA)

#Developed by Ronald Fisher the father of modern statistics. LDA is a classifier. Could bring better results than logitic regression. Supervised classfication.

from sklearn.lda import LDA

lda_2c = LDA(n_components=2)

X_lda_2c = lda_2c.fit_transform(iris.data, iris.target)

plt.scatter(X_lda_2c[:,0], X_lda_2c[:,1], c=iris.target, alpha=0.8, edgecolors='none')

plt.show()

#%%

#Latent Semantical Analysys (LSA)

#Aplies SVD to the dataseet, creating a set of words usually associated with the same concept. LSA used when all the words are in the documents and are presented in large numbers.

from sklearn.datasets import fetch_20newsgroups

categories = ['sci.med', 'sci.space']

twenty_sci_news = fetch_20newsgroups(categories=categories)

from sklearn.feature_extraction.text import TfidfVectorizer

tf_vect = TfidfVectorizer()

word_freq = tf_vect.fit_transform(twenty_sci_news.data)

from sklearn.decomposition import TruncatedSVD

tsvd_2c = TruncatedSVD(n_components=50)

tsvd_2c.fit(word_freq)

np.array(tf_vect.get_feature_names())[tsvd_2c.components_[20].argsort()[-10:][::-1]]

#%%

#Independent Component Analysis (ICA)

#ICA is a technique that allows you to create maximally independent additive subcomponents from the initial multivariate input signa

#Lot of applications in neurological data and neuroscience

#use sklearn.decomposition.FastICA

#%%

#Kernel PCA

#create a circular dataset to show the Kernel PCA
def circular_points(radius, N):
	return np.array([[np.cos(2*np.pi*t/N)*radius, np.sin(2*np.pi*t/N)*radius] for t in range(N)])

N_points = 50

fake_circular_data = np.vstack([circular_points(1.0, N_points), circular_points(5.0, N_points)])

fake_circular_data += np.random.rand(*fake_circular_data.shape)

fake_circular_target = np.array([0]*N_points + [1]*N_points)

plt.scatter(fake_circular_data[:,0], fake_circular_data[:,1], c=fake_circular_target, alpha=0.8, s=60, marker='o', edgecolors='white')
plt.show()

#%%

#Due to the circular feature of the data, all linear transformation would fail to separate the data.

from sklearn.decomposition import KernelPCA

kpca_2c = KernelPCA(n_components=2, kernel='rbf')

X_kpca_2c = kpca_2c.fit_transform(fake_circular_data)

plt.scatter(X_kpca_2c[:,0], X_kpca_2c[:,1], c=fake_circular_target, alpha=0.8, s=60, marker='o', edgecolors='white')
plt.show()

#Now after the Kernel PCA transformation we can use linear techniques

#%%

#T-SNE

# The core of the algorithm is based on two rules: the first is that recurrent similar observations must have a greater contribution to the output (achieved with a probability distribution function); second, the distribution in the high dimensional space must be similar to the one in the small space (achieved minimized the KL, Kullback-Leibler, divergence between the two probability distribution functions). 

#Apply T-SNE to the iris dataset

from sklearn.manifold import TSNE

from sklearn.datasets import load_iris

iris = load_iris()

X, y = iris.data, iris.target
X_tsne = TSNE(n_components=2).fit_transform(X)

plt.scatter(X_tsne[:,0], X_tsne[:, 1], c=y, alpha=0.8, s=60, marker='o', edgecolors='white')

plt.show()
						
#%%

#Restricted Boltzmann Machine (RBM)

#The main hypothesis of this technique is that the input dataset is composed of features that represent probability (binary values or real values in the [0,1] range) since RBM is a probabilistic approach.

from sklearn import preprocessing
from sklearn.neural_network import BernoulliRBM

n_components = 144 #Try with 64, 100, 144

olivetti_faces = datasets.fetch_olivetti_faces()

X = preprocessing.binarize(preprocessing.scale(olivetti_faces.data.astype(float)), 0.5)

rbm = BernoulliRBM(n_components=n_components, learning_rate=0.01, n_iter=100)

rbm.fit(X)

plt.figure(figsize=(4.2, 4))

for i, comp in enumerate(rbm.components_):
	plt.subplot(int(np.sqrt(n_components+1)), int(np.sqrt(n_components+1)), i+1)

plt.imshow(comp.reshape((64, 64)), cmap=plt.cm.gray_r, interpolation='nearest')

plt.xticks(())

plt.yticks(())

plt.suptitle(str(n_components) + ' components extracted by RBM', fontsize=16)

plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()