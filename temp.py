# coding=<utf-8>

import numpy as np
import pandas as pd
import re
from konlpy.tag import Komoran
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, LSTM 
from keras.models import Sequential 
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
max_len = 30

X_train = []
Y_train = []
X_test = []
Y_test =  []
train_data = pd.read_csv("dataset.csv", encoding='UTF8' )
train_data = train_data.dropna()
train_data = shuffle(train_data)
stopwords = ['.', ',', '~']

# 토크나이징 및 인코딩
mc = Komoran()
for sentence in train_data['Sentence']:
    temp_X = []
    sentence.replace('.','').replace(',','').replace('~','').replace('"','')
    temp_X = mc.morphs(sentence)
    X_train.append(temp_X)
    X_test.append(temp_X)

tokenizer = Tokenizer(35000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
#Y_train = labeling(train_data, Y_train)
X_test = tokenizer.texts_to_sequences(X_test)
#Y_test = labeling(train_data, Y_test)

threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
vocab_size = total_cnt - rare_cnt + 1 # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
print('단어 집합의 크기 :',vocab_size)
