# -*- coding: utf-8 -*-
"""
Created on Thu May  3 09:38:27 2018

BA_05032018
Text Mining Filter

@author: Justin
"""

import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week9\\gender-classifier-DFE-791531.csv', encoding = 'latin1')

data['gender'].value_counts() # brand는 marketing ID를 의미

bin_data = data[data['gender'].isin(['female', 'male'])] # 이 클래스에 속하는 애들만 선택할 수 있다.

text = bin_data['text']
text.head() # unrelevant term이 있다. ex) retweet...

# regular expression 을 쓰고 싶다면 re를 import 하자
import re

# we have to clean text one by one

url_pattern = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})'
sample = ''

clean = text.apply(lambda x: re.sub('#\w+', '', x)) # '#' 으로 시작하는 단어를 지움
clean = clean.apply(lambda x: re.sub('@\w+', '', x))
clean = clean.apply(lambda x: re.sub(url_pattern, '', x))

# certain character not in Alphabet
import string # provide information of the many different character using a language.

string.punctuation
string.ascii_letters # non-Alphabet을 지울 수 있음

clean = clean.apply(lambda x: ''.join([y for y in x if not y in string.punctuation]))
clean.head()
clean = clean.apply(lambda x: ''.join([y for y in x if y in string.ascii_letters+' '])) # keep the space 하고 싶으면 ' ' 넣자.

                                      
clean = clean.apply(lambda x: x.lower())

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df = 100) # min_df는 53라인 후에 추가했음

cv.fit(clean) # there is no target, just converting. It just scan for defining the features.

vocab = cv.vocabulary_ # scan 후 vocabulary를 만듦 , 하지만 너무 feature가 많음. 그래서 min_df를 설정해야함
# 이후 stemming을 해서 품사 조정된 true relevant feature를 얻을 수 있다.
# term frequenct matrix에 적용해보자

X = cv.transform(clean) # variable explorer에서 볼 수 없는데, 왜냐하면 얘는 sparse matrix -> many elements are zero.
type(X)
X = X.toarray() 

tokenizer = cv.build_tokenizer()
tokenizer(clean.values[0])


from sklearn.feature_extraction.text import TfidfTransformer

trans = TfidfTransformer(sublinear_tf = True)
Xtfidf = trans.fit_transform(X)
Xtfidf = Xtfidf.toarray() # 이걸 적용하니까 integer 넘버가 안 나옴

idf = trans.idf_ # 숫자가 낮은게 많이 사용되었다.

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

fs = SelectKBest(score_func = chi2, k = 100)
y = (bin_data['gender'] == 'female')*1

fs.fit(X, y) # Target이 필요해, Xtfidf대신에 X 넣어야함 count value가 필요하게 때문에 카이스퀘어는
Xred = fs.transform(X)
fs.get_support(True)

chi2_val = chi2(X,y) # 0은 chi2 statstic value, 1은 probablity value (p-value)
# small p-value accept alternative hypothesis

ind = np.argsort(chi2_val[0])[::-1] # we can get the index, 어떤위치에 있는지, sort는 안됨 Ascending 한 index를 얻기위해
vocab = pd.DataFrame([(x,y) for x,y in vocab.items()])

sel_vocab = vocab[vocab[1].isin(ind[:100])] # quite not independent on gender


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(Xred,y)

p = np.exp(mnb.feature_log_prob_) 

mnb.score(Xred, y)

# 단지 raw data에서 processing 하는 법을 보여주기 위해




###practice

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
corpus = ['This is the first document.','This is the second second document.', 'And the third one.', 'Is this the first document?']
X = vectorizer.fit_transform(corpus)
