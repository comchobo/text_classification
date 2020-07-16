import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential 
from keras.layers import *

from keras.models import load_model
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from keras_self_attention import SeqSelfAttention
import random

max_len = 30


# 1. 데이터셋 준비와 변수 정의
X_data = []
Y_data = []
train_data = pd.read_csv("dataset3.csv", encoding='utf-8',sep='|')
train_data = train_data.dropna()
train_data = shuffle(train_data)
k=0
stopwords = ['의','가','이','은','들','는','과','도','를','으로','에','와']

# 2. 형태소 분석, 각종 특수기호와 숫자 제거

mc = Okt() #okt 형태소 분석기를 사용하였습니다.
for sentence in train_data['Sentence']:
    temp_X = []
    sentence = re.sub('\,|~|\"|=|<|>|\*|\'', '', sentence)
    sentence = re.sub('\(|\)', ',', sentence)
    sentence = re.sub('[0-9]+', 'num', sentence)
    sentence = re.sub(";+", ';', sentence)
    sentence = re.sub("[?]{2,}", '??', sentence)
    sentence = re.sub("[.]{2,}", '..', sentence)
    sentence = re.sub("[!]{2,}", '!!', sentence)
    #sentence = re.sub('[a-zA-Z]', '', sentence)
    temp_X = mc.morphs(sentence, norm=True,stem=True)
    temp_X = [word for word in temp_X if not word in stopwords] 
    X_data.append(temp_X)

vocab_size=25000

tokenizer0 = Tokenizer(vocab_size, oov_token = 'OOV',filters='')#temp.py에서 얻은 수치입니다.
tokenizer0.fit_on_texts(X_data)

# 3.  토큰 수 제한
threshold = 2
total_cnt = len(tokenizer0.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer0.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
        
vocab_size = total_cnt - rare_cnt + 1

# 4. 레이블 원핫인코딩 및 토크나이저 정의
print(vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token = 'OOV',filters='')
tokenizer.fit_on_texts(X_data)

X_data = tokenizer.texts_to_sequences(X_data)

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

Y_data = labeling(train_data, Y_data)



#5. Bootstrap 기법을 위한 함수 정의

def res(X_data,Y_data):
    X_data, Y_data = shuffle(X_data, Y_data)
    
    X_train = []
    Y_train = []
    OutOfBag=[]
    selected=[]
    X_test = []
    Y_test = []
    Z = [i for i in range(26965)]
    for i in range(30000):
        index = random.choice(Z)
        X_train.append(X_data[index])
        Y_train.append(Y_data[index])
        selected.append(index)
        
    my_set = set(selected)
    selected = list(my_set)
    OutOfBag = [item for item in Z if item not in selected]
    
    for index in OutOfBag:
        X_test.append(X_data[index])
        Y_test.append(Y_data[index])
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    

    X_train = pad_sequences(X_train, maxlen=max_len) #문장길이 30으로 제한
    X_test = pad_sequences(X_test, maxlen=max_len)
    return X_test,Y_test,X_train,Y_train

#6. 토크나이저 저장

import json
json = json.dumps(tokenizer.word_index)
f3 = open("wordIndexOkt_test5.json", "w")
f3.write(json)
f3.close()

#7. LSTM Model 구현부 (10개의 bootstrap)

X_test,Y_test,X_train,Y_train = res(X_data,Y_data)
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens1_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)


X_test,Y_test,X_train,Y_train = res(X_data,Y_data)
model1 = Sequential()
model1.add(Embedding(vocab_size, 128, input_length=max_len))
model1.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model1.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model1.add(Flatten())
model1.add(Dense(5, activation='softmax'))
print(model1.summary())
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens2_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model1.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)

X_test,Y_test,X_train,Y_train = res(X_data,Y_data)
model2 = Sequential()
model2.add(Embedding(vocab_size, 128, input_length=max_len))
model2.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model2.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model2.add(Flatten())
model2.add(Dense(5, activation='softmax'))
print(model2.summary())
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens3_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model2.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)


X_test,Y_test,X_train,Y_train = res(X_data,Y_data)
model3 = Sequential()
model3.add(Embedding(vocab_size, 128, input_length=max_len))
model3.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model3.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model3.add(Flatten())
model3.add(Dense(5, activation='softmax'))
print(model3.summary())
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens4_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model3.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)


X_test,Y_test,X_train,Y_train = res(X_data,Y_data)
model4 = Sequential()
model4.add(Embedding(vocab_size, 128, input_length=max_len))
model4.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model4.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model4.add(Flatten())
model4.add(Dense(5, activation='softmax'))
print(model4.summary())
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens5_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model4.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)

X_test,Y_test,X_train,Y_train = res(X_data,Y_data)
model5 = Sequential()
model5.add(Embedding(vocab_size, 128, input_length=max_len))
model5.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model5.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model5.add(Flatten())
model5.add(Dense(5, activation='softmax'))
print(model5.summary())
model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens6_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model5.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)

X_test,Y_test,X_train,Y_train = res(X_data,Y_data)

model6 = Sequential()
model6.add(Embedding(vocab_size, 128, input_length=max_len))
model6.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model6.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model6.add(Flatten())
model6.add(Dense(5, activation='softmax'))
print(model6.summary())
model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens7_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model6.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)

X_test,Y_test,X_train,Y_train = res(X_data,Y_data)

model7 = Sequential()
model7.add(Embedding(vocab_size, 128, input_length=max_len))
model7.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model7.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model7.add(Flatten())
model7.add(Dense(5, activation='softmax'))
print(model7.summary())
model7.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens8_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model7.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)

X_test,Y_test,X_train,Y_train = res(X_data,Y_data)

model8 = Sequential()
model8.add(Embedding(vocab_size, 128, input_length=max_len))
model8.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model8.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model8.add(Flatten())
model8.add(Dense(5, activation='softmax'))
print(model8.summary())
model8.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens9_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model8.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)

X_test,Y_test,X_train,Y_train = res(X_data,Y_data)

model9 = Sequential()
model9.add(Embedding(vocab_size, 128, input_length=max_len))
model9.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model9.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       attention_activation='sigmoid',
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention'))
model9.add(Flatten())
model9.add(Dense(5, activation='softmax'))
print(model9.summary())
model9.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('Project_model_Ens10_.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = model9.fit(X_train, Y_train, epochs=15, callbacks=[es, mc], batch_size=30, validation_data=(X_test, Y_test), verbose=1)


print("model saved!")
