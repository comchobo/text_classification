# coding=<utf-8>
import pandas as pd
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences

def put(token, sentence, mc):  
        sentence2 = []
        sentence.replace('.','').replace(',','').replace('~','').replace('"','')
        sentence2 = mc.morphs(sentence)
        if len(sentence2)<2:
            output= token.texts_to_sequences(sentence)
            output=pad_sequences(output, 30)
        else:
            output= token.texts_to_sequences(sentence2)
            output=pad_sequences(output, 30)
        output = np.array(output)
        return output

# 결과 변환 함수
def convert(input):
    result = np.argmax(input, axis=1)
    return result

def out(sentence, model) :
    #기본처리
    final=np.array([])

    try:
        k=0
        for nums in sentence:
            if any(nums) == False:
                   continue
            final=np.append(final,nums)
            k=k+1
        final2=np.array([])
        final2=np.append(final2,final)
        final2=final2.reshape(k,30)
        return convert(model.predict(final2))

    except:
        return [5]
