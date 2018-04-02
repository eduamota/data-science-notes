# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:26:53 2018

@author: emota
"""

#%%

import pandas as pd

categorical_feature = pd.Series(['sunny', 'cloudy', 'snowy', 'rainy', 'foggy'])

mapping = pd.get_dummies(categorical_feature)

mapping

#%%

from sklearn.datasets import fetch_20newsgroups

categories = ['sci.med', 'sci.space']

twenty_sci_news = fetch_20newsgroups(categories=categories)

#%%

print(twenty_sci_news.data[0])

#%%

twenty_sci_news.filenames

#%%

print (twenty_sci_news.target[0])
print(twenty_sci_news.target_names[twenty_sci_news.target[0]])

#%%

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

word_count = count_vect.fit_transform(twenty_sci_news.data)

word_count.shape

#%%

word_list = count_vect.get_feature_names()

for n in word_count[0].indices:
	print('Word "%s" appears %i times' % (word_list[n], word_count[0,n]))
	
#%%

from sklearn.feature_extraction.text import TfidfVectorizer

#The sum of the frequencies is 1 (or close to 1 due to the approximation). This happens because we chose the l1 norm. In this specific case, the word frequency is a probability distribution function. Sometimes, it's nice to increase the difference between rare and common words. In such cases, you can use the l2 norm to normalize the feature vector.

tf_vect = TfidfVectorizer(use_idf=False, norm='l1')

word_freq = tf_vect.fit_transform(twenty_sci_news.data)

word_list = tf_vect.get_feature_names()

for n in word_freq[0].indices:
	print('Word "%s" has frequency %0.3f' % (word_list[n], word_freq[0, n]))

#%%

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer() # Default: use_idf=True
word_tfidf = tfidf_vect.fit_transform(twenty_sci_news.data)
word_list = tfidf_vect.get_feature_names()
for n in word_tfidf[0].indices:
	print ('Word "%s" has tf-idf %0.3f' % (word_list[n],
        word_tfidf[0, n]))

#%%

text_1 = 'we love data science'
text_2 = 'data science is hard'
documents = [text_1, text_2]
documents


#%%

# That is what we say above, the default one
count_vect_1_grams = CountVectorizer(ngram_range=(1, 1),
stop_words=[], min_df=1)
word_count = count_vect_1_grams.fit_transform(documents)
word_list = count_vect_1_grams.get_feature_names()
print ("Word list = ", word_list)
print ("text_1 is described with", [word_list[n] + "(" +
str(word_count[0, n]) + ")" for n in word_count[0].indices])
	
#%%

# Now a bi-gram count vectorizer
count_vect_1_grams = CountVectorizer(ngram_range=(2, 2))
word_count = count_vect_1_grams.fit_transform(documents)
word_list = count_vect_1_grams.get_feature_names()
print ("Word list = ", word_list)
print ("text_1 is described with", [word_list[n] + "(" +
str(word_count[0, n]) + ")" for n in word_count[0].indices])
	
#%%

 # Now a uni- and bi-gram count vectorizer
count_vect_1_grams = CountVectorizer(ngram_range=(1, 2))
word_count = count_vect_1_grams.fit_transform(documents)
word_list = count_vect_1_grams.get_feature_names()
print ("Word list = ", word_list)
print ("text_1 is described with", [word_list[n] + "(" +
str(word_count[0, n]) + ")" for n in word_count[0].indices])
	
#%%

from sklearn.feature_extraction.text import HashingVectorizer
hash_vect = HashingVectorizer(n_features=1000)
word_hashed = hash_vect.fit_transform(twenty_sci_news.data)
word_hashed.shape


#%%

'''
Financial institutions scrape the Web to extract fresh details and information about the companies in their portfolio. Newspapers, social networks, blogs, forums, and corporate websites are the ideal targets for these analyses.

Advertisement and media companies analyze sentiment and the popularity of many pieces of the Web to understand people's reactions.

Companies specialized in insight analysis and recommendation scrape the Web to understand patterns and model user behaviors.

Comparison websites use the web to compare prices, products, and services, offering the user an updated synoptic table of the current situation.
'''
#%%
import requests

url = 'https://en.wikipedia.org/wiki/William_Shakespeare'

req = requests.get(url)

response = req.text

#%%

from bs4 import BeautifulSoup

soup = BeautifulSoup(response, 'html.parser')

#%%

soup.title

#%%

section = soup.find_all(id='mw-normal-catlinks')[0]

for catlink in section.find_all("a")[1:]:
	print(catlink.get("title"), "->", catlink.get("href"))