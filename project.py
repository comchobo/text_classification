import numpy as np
import pandas as pd
import re
from konlpy.tag import Komoran
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential 
from keras.layers import *
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

import sys
sys.path.insert(0, '../input/attention')
from keras_self_attention import SeqSelfAttention

max_len = 30

def labeling(data, train):
    for i in range(len(data['Emotion'])):
        if data['Emotion'].iloc[i] == '슬픔':
            train.append([1, 0, 0, 0, 0])
        elif data['Emotion'].iloc[i] == '중립':
            train.append([0, 1, 0, 0, 0])
        elif data['Emotion'].iloc[i] == '행복':
            train.append([0, 0, 1, 0, 0])
        elif data['Emotion'].iloc[i] == '공포':
            train.append([0, 0, 0, 1, 0])
        elif data['Emotion'].iloc[i] == '분노':
            train.append([0, 0, 0, 0, 1])
    return train

#데이터셋 준비와 변수 정의
X_train = []
Y_train = []
X_test = []
Y_test =  []
train_data = pd.read_csv("dataset.csv", encoding='UTF8' )
train_data = train_data.dropna()
train_data = shuffle(train_data)
k=0

# 토크나이징 및 인코딩
mc = Komoran() #코모란 형태소 분석기를 사용하였습니다.
for sentence in train_data['Sentence']:
    temp_X = []
    sentence.replace('.','').replace(',','').replace('~','').replace('"','') #각종 잡다한 특수문자를 제거합니다.
    temp_X = mc.morphs(sentence)
    X_test.append(temp_X)
    X_train.append(temp_X)
    
tokenizer = Tokenizer(9482) #temp.py에서 얻은 수치입니다.
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
Y_train = labeling(train_data, Y_train)
X_test = tokenizer.texts_to_sequences(X_test)
Y_test = labeling(train_data, Y_test)

#모델에 넣기 위해 행렬화
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

#토큰 하나만 있는 행은 제거합니다.
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
drop_test = [index for index, sentence in enumerate(X_test) if len(sentence) < 1]
X_train = np.delete(X_train, drop_train, axis=0)
Y_train = np.delete(Y_train, drop_train, axis=0)
X_test = np.delete(X_test, drop_test, axis=0)
Y_test = np.delete(Y_test, drop_test, axis=0)

#토크나이저 저장
import json
json = json.dumps(tokenizer.word_index)
f3 = open("wordIndex.json", "w")
f3.write(json)
f3.close()

#LSTM Model 구현부
X_train = pad_sequences(X_train, maxlen=max_len) #문장길이 30으로 제한
X_test = pad_sequences(X_test, maxlen=max_len)
model = Sequential()
model.add(Embedding(9482, 128, input_length=max_len))
model.add(Bidirectional(LSTM(units=156, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
history = model.fit(X_train, Y_train, epochs=4, batch_size=15, validation_split=0.1)

#만들어진 모델 저장
from keras.models import load_model
model.save('Project_model.h5')
print("model saved!")
