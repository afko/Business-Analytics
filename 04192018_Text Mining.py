# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:03:53 2018

BA_04192018
Text Mining

@author: Justin
"""

import nltk

nltk.download()
# stopwords collection은 usually 텍스트마이닝에서 제외하는 단어를 모아놨다.
# nltk lemmatization 은 worknet을 기반으로 분석한다.

from nltk.corpus import gutenberg
ids = gutenberg.fileids() # 옛것이라 라이센스 없어서 괜찮음 ㅎ

text = gutenberg.open(ids[0]).read() # emma 로 분석을 시작해보자.

nltk.download('punkt')
from nltk import word_tokenize
tokens = word_tokenize(text)
tokens[:100]

en = nltk.Text(tokens)
#tokens = en.tokens # 모든 character를 나눈다. nltk.Text에 text를 넣으면.
dic = en.vocab()
en.plot(50)

lower_tokens = [x.lower() for x in tokens] # 모든 character를 lower case로.
en_lw = nltk.Text(lower_tokens)
dic_lw = en_lw.vocab()

words = list(dic_lw.keys())

# practice page: 9
en.concordance('Emma', lines = 5) # concordance는 그 용어가 사용된 곳을 보여주는 용어 색인이다.
en.similar('Emma') # frequency로 판별 앞뒤 맥락을 이용하여
en.collocations() # default값으로 몇개가 출력되는지 설정되어있다. (20)

# practice page: 10
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('universal_tagset')

from nltk import pos_tag

sent = 'I am not there.'
words = word_tokenize(sent)
tagged =  pos_tag(words, tagset = 'universal')

#practice page: 13,14
'12121dcddd'.isalnum()
'23$dfdf'.isalnum()

#practice page: 15
from nltk.corpus import stopwords

stopwords.words('english') 

#practice page: 16, 17, 18
nltk.download('porterstemmer') 
from nltk.stem.porter import PorterStemmer # Rule Base이기 때문에 시간이 얼마 걸리지 않는다.

pstem = PorterStemmer()
pstem.stem('maximum')
pstem.stem('studying')
pstem.stem('owed')
pstem.stem('saying')

from nltk.stem.lancaster import LancasterStemmer

lstem = LancasterStemmer()
lstem.stem('owed') # 이 경우에는 ed가 없어서 od가 된다. (...)
lstem.stem('stduying')

from nltk.stem.snowball import SnowballStemmer

sstem = SnowballStemmer('english')
sstem.stem('studying')
sstem.stem('owed')


#practice page: 19

from nltk.stem import WordNetLemmatizer # 시간이 더 걸린다.
wordnet = WordNetLemmatizer()

wordnet.lemmatize('dogs')
wordnet.lemmatize('studying') # default pos는 NOUN이다.
wordnet.lemmatize('studying', pos = 'v')

