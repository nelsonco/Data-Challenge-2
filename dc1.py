# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 09:23:59 2020

@author: Courtney
"""
import random
from itertools import chain
import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize
import string
import matplotlib.pyplot as plt

# Preping the Data
df = pd.read_excel('spam 2.xlsx', sheet_name='Sheet1')
df = df.drop('Unnamed: 2', axis=1)
df = df.drop('Unnamed: 3', axis=1)
df = df.drop('Unnamed: 4', axis=1)
df['v3'] = df['v2'].str.replace('[!,\n,\t,),?,.,",-,...,&,;,#,_,*,:,/,%,(,$,\,+]',' ')
df = df.drop('v2', axis=1)

# Separating Spam
spam = df.loc[df['v1'] == 'spam']
print(len(spam))
#print(spam)

#Separating Ham
ham = df.loc[df['v1'] == 'ham']
#print(ham)


""" HAM """
ham.dropna(inplace=True)

ham_tags = []
ham_tokens = []
for sentences in ham.iloc[:, 1]:
    #print(sentences)
    tokens = word_tokenize(sentences)
    tags = nltk.pos_tag(tokens)
    #print(tags)
    ham_tags.append(tags)
    for tag in tags:
        if (tag[1] == 'JJ' or tag[1] == 'NN'  ):
            ham_tokens.append(tag[0])


ham_tags = list(chain.from_iterable(ham_tags))
#print(ham_tags)
plt.ion()
hamfq = nltk.FreqDist(ham_tokens)
plt.figure(figsize=(30, 20))  # the size you want
hamfq.plot(30, cumulative=False)
plt.savefig('img_top10_common.png')
plt.ioff()
plt.show()

""" SPAM """
spam.dropna(inplace=True)

spam_tags = []
spam_tokens = []
for sentences in spam.iloc[:, 1]:
    #print(sentences)
    tokens = word_tokenize(sentences)
    tags = nltk.pos_tag(tokens)
    #print(tags)
    spam_tags.append(tags)
    for tag in tags:
        if (tag[1] == 'JJ' or tag[1] == 'NN'  ):
            spam_tokens.append(tag[0])


spam_tags = list(chain.from_iterable(ham_tags))
#print(spam_tags)
plt.ion()
hamfq = nltk.FreqDist(spam_tokens)
plt.figure(figsize=(30, 20))  # the size you want
hamfq.plot(30, cumulative=False)
plt.savefig('spam.png')
plt.ioff()
plt.show()
#print(ham_tags)

#print('spam \n')
#print(wordFrequency(spam).most_common())

"""
Sort by pre-determined key words
"""
df.dropna(inplace = True)
spam_key_words = ['www','prize', 'subscribe', 'subscribed','sex', 'claim',
                   'awarded', 'award', 'cash', 'SMS']

spam_manual = []
ham_manual = []

count = 0
for sentence in df.iloc[:, 1]:
    indicator = 0 
    #print(sentences)
    tokens = word_tokenize(sentence)
    for token in tokens:
        #print(token)
        if token in spam_key_words:
            spam_manual.append([df.iloc[count,0],sentence])
            indicator = 1
            break
    if indicator == 0:
        ham_manual.append(sentence)
    count = count+1

count = 0
for sentence in spam_manual:
    if sentence[0] == 'spam':
        count = count + 1
    #print(sentence)
#print(count)
#print(len(spam))

"""
Training Set
"""

# HAM TRAINING AND TESTING SETS
ham_testing = []
for sentence in ham.iloc[:,1]:
    ham_testing.append(sentence)

ham_training = []
ham_size = int(.1*len(ham))
i = 0
index_list = []
for i in range (0,ham_size):
    index = random.randint(0,len(ham)-1)
    while index in index_list:
        index = random.randint(0,len(ham)-1)
    index_list.append(index)
    ham_training.append(ham.iloc[index,1])
    ham_testing.remove(ham.iloc[index,1])
    i = i+1
    

# FINDING FREQUENCY DISTRIBUTION OF TRAINING SET
ham_training_tokens = []
ham_training_tags = []
for sentences in ham_training:
    #print(sentences)
    tokens = word_tokenize(sentences)
    tags = nltk.pos_tag(tokens)
    #print(tags)
    ham_training_tags.append(tags)
    for tag in tags:
        if (tag[1] == 'JJ' or tag[1] == 'NN'  ):
            ham_training_tokens.append(tag[0])
ham_training_distribution = nltk.FreqDist(ham_training_tokens)
ham_dist = ham_training_distribution.most_common()
            

# SPAM TRAINING AND TESTING SETS
spam_testing = []
for sentence in spam.iloc[:,1]:
    spam_testing.append(sentence)


spam_training = []
spam_size = int(.1*len(spam))
i = 0
index_list = []
for i in range (0,spam_size):
    index = random.randint(0,len(spam)-1)
    while index in index_list:
        index = random.randint(0,len(spam)-1)
    index_list.append(index)
    spam_training.append(spam.iloc[index,1])
    spam_testing.remove(spam.iloc[index,1])
    i = i+1

    
spam_training_tokens = []
spam_training_tags = []
for sentences in spam_training:
    #print(sentences)
    tokens = word_tokenize(sentences)
    tags = nltk.pos_tag(tokens)
    #print(tags)
    spam_training_tags.append(tags)
    for tag in tags:
        if (tag[1] == 'JJ' or tag[1] == 'NN'  ):
            spam_training_tokens.append(tag[0])
spam_training_distribution = nltk.FreqDist(spam_training_tokens)
spam_dist = spam_training_distribution.most_common()


"""
Each sentence will get a ham and a spam score and the score will determine which
category the word falls into
"""     
spam_score = 0
ham_score = 0  
test = ham_testing
for sentence in spam_testing:
    test.append(sentence)     

for sentence in test:
   tokens = word_tokenize(sentence)
   for token in tokens:
     for i in ham_dist:
       if token in i[0]:
           ham_score = ham_score + i[1]
        
        
ham_score = ham_score/len(ham_dist)
print(ham_score)



"""
for i,j in smsData.iterrows():
    #print(j)
   # print(smsData)
    if(v1 == 'spam'):
        print('yes')
    if smsData.index[i] == 'spam':
        spam.append(smsData.index[i,2])

print(spam)
"""
#smsSpam = pd.read_csv('sms_Spam - Sheet1.csv')
#smsSpam = smsSpam.drop(columns=['Unnamed: 2', 'Unnamed: 3'])

"""
def wordFrequency(text_dataframe):
    wordFreq = nltk.FreqDist()

    for sentences in text_dataframe.iloc[:, 1]:
        tokenized = sentences.split(" ")
        for word in tokenized:
            wordFreq[word.lower()] += 1
    return wordFreq

spam_wordFreq = wordFrequency(smsData)

ham_wordFreq = wordFrequency(smsData)

#classifier = nltk.classify.NaiveBayesClassifier.train()


"""
