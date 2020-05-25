# coding=<utf-8>
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import re
from konlpy.tag import Komoran
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential 
from tensorflow.keras import layers
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import sys
sys.path.insert(0, '../input/attention')
from keras_self_attention import SeqSelfAttention
import source as s
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

X_train = []
Y_train = []
X_test = []
Y_test =  []
train_data = pd.read_csv("dataset.csv", encoding='UTF8' )
train_data = train_data.dropna()
train_data = shuffle(train_data)
stopwords = ['.', ',', '~', '"']
k=0

# 토크나이징 및 인코딩
mc = Komoran()
for sentence in train_data['Sentence']:
    temp_X = []
    sentence.replace('.','').replace(',','').replace('~','').replace('"','')
    temp_X = mc.morphs(sentence)
    X_test.append(temp_X)
    X_train.append(temp_X)

tokenizer = Tokenizer(9482)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
Y_train = labeling(train_data, Y_train)
X_test = tokenizer.texts_to_sequences(X_test)
Y_test = labeling(train_data, Y_test)

#인코딩 후 저장
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
drop_test = [index for index, sentence in enumerate(X_test) if len(sentence) < 1]
X_train = np.delete(X_train, drop_train, axis=0)
Y_train = np.delete(Y_train, drop_train, axis=0)
X_test = np.delete(X_test, drop_test, axis=0)
Y_test = np.delete(Y_test, drop_test, axis=0)
#my_title_dic = {'Sentence':X_train, 'Emotion': Y_train}
#my_title_df = pd.DataFrame.from_dict(my_title_dic)
#my_title_df.to_csv("test.csv")

#토크나이저 저장
import json
json = json.dumps(tokenizer.word_index)
f3 = open("wordIndex.json", "w")
f3.write(json)
f3.close()

#LSTM Model 구현부
X_train = pad_sequences(X_train, maxlen=max_len) #문장길이 조절
X_test = pad_sequences(X_test, maxlen=max_len)

inputs = layers.Input(shape=(max_len,))
embedding_layer = s.TokenAndPositionEmbedding(30, 9482, 32)
x = embedding_layer(inputs)
transformer_block = s.TransformerBlock(32, 2, 32)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, Y_train, batch_size=32, epochs=2, validation_split=0.1)

'''
model = Sequential()
model.add(Embedding(9482, 128, input_length=max_len))
model.add(Bidirectional(LSTM(units=156, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
'''
'''
#CNN Model 구현부
model = Sequential()
model.add(Embedding(9482,128,input_length=max_len))
model.add(Dropout(0.2))
model.add(Conv1D(80,5,activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(45, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
'''

#history = model.fit(X_train, Y_train, epochs=4, batch_size=15, validation_split=0.1)

#직접 테스트
#predict = model.predict(X_test)
#import numpy as np
#predict_labels = np.argmax(predict, axis=1)
scores = model.evaluate(X_test, Y_test, verbose=0)
# 텍스트 데이터에 대해서 정확도 평가
print('정확도 : %.2f%%' % (scores[1]*100))

#만들어진 모델 저장
from keras.models import load_model
model.save('Project_model.h5')
print("model saved!")
