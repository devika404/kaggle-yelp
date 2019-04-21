try:
    import urllib.request as urllib2
except ImportError:
    import urllib2 
from bs4 import BeautifulSoup
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from pprint import pprint
import re

#YELP API call
YELP_TOKEN = "Vsxrldjcj5BNhekekeisVcBzzOTNbjlb_I7Mq9I38ca3rkjhoJvOJQYFXv3DBK1Rdgj-Cwgjej16kopYPb8cvDQfFFwnvFSyLVb-TTotUklUSvjXeFZBtZF1j3sVXHYx"
r = requests.get("https://api.yelp.com/v3/businesses/search?location=Toronto&limit=50", headers={"Authorization": "Bearer %s" % YELP_TOKEN})
r.json()

#Getting url of restaurants
review_labels = []
for business in r.json()['businesses']:
    reviews = requests.get("https://api.yelp.com/v3/businesses/%s/reviews" % business['id'], headers={"Authorization": "Bearer %s" % YELP_TOKEN}).json()
    for review in reviews['reviews']:
        review_labels.append(review['url'])
review_labels

a = len(review_labels)

ratings =[]
reviews =[]
reviews_label=[]
    

#Getting 100 reviews for each restaurant
for i in range(len(review_labels)):
    url2 = review_labels[i]
    start = 0
    num_pages = 2
    end = 20* num_pages
    
    
    while (start < end):
        url = url2 + '?start=' + str(start)
        start +=20
        print(url)
    
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page,"lxml")
    
        
        for reviewBody in soup.findAll('div',{"class":"review-content"}):
            ratings.append((reviewBody.div.div.div.get("title")))
            reviews.append((reviewBody.find('p').text))
            reviews_label.append((reviewBody.find('p').text, int(reviewBody.div.div.div.get("title")[0])))
            
print(ratings)
print(reviews)
print(reviews_label)

#Writting dataset as json file
reviews_jfile = reviews_label
reviews_jfile


review_jfeatures = [(x, 'positive' if y > 3 else 'negative') for (x, y) in reviews_jfile]
review_jfeatures

flist = [review_jfeatures]

data = json.dumps(flist)
print(data)

#Splitting reviews with rating greater than 3 as positive and rest as negative
review_features = [(x.split(' '), 'positive' if y > 3 else 'negative') for (x, y) in reviews_label]
review_features

#Classification
from nltk.sentiment import SentimentAnalyzer
import nltk.sentiment.util
from nltk.classify import NaiveBayesClassifier

b = len(reviews_label)
c = int(b*0.7)

random.shuffle(review_features)
training_docs = review_features[:c]
test_docs = review_features[c:]

print("Training: %d, Testing: %d" % (len(training_docs), len(test_docs)))

sentim_analyzer = SentimentAnalyzer()

all_words_neg = sentim_analyzer.all_words([nltk.sentiment.util.mark_negation(doc) for doc in training_docs])
all_words_neg

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(test_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
     print('{0}: {1}'.format(key, value))
     