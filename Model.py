# coding=<utf-8>
import pandas as pd
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences

def put(token, sentence, mc):  
        sentence2 = []
        sentence.replace('.','').replace(',','').replace('~','').replace('"','') #문장의 잡다한 기호 제거
        sentence2 = mc.morphs(sentence)
        if len(sentence2)<2: #만약 문장이 토큰 단 하나로 구성되어있을 경우
            output= token.texts_to_sequences(sentence) #형태소 분석기 미사용
            output=pad_sequences(output, 30)
        else:
            output= token.texts_to_sequences(sentence2)
            output=pad_sequences(output, 30) #그 외엔 정상적으로 분석
        output = np.array(output) #모델에 넣기 위해 행렬로 변환
        return output

# 결과 변환 함수
def convert(input):
    result = np.argmax(input, axis=1)
    return result
# predict 함수
def out(sentence, model) :
    #기본처리
    final=np.array([])

    try:
        k=0
        for nums in sentence:
            if any(nums) == False:
                   continue #단어가 wordindex(학습한 단어 내용)에 없을 경우 건너뜀
            final=np.append(final,nums) # 그 외에는 final 행렬에 삽입
            k=k+1
        final2=np.array([])
        final2=np.append(final2,final)
        final2=final2.reshape(k,30) # 모델에 넣기 위해 final을 재조립
        return convert(model.predict(final2))

    except:
        return [5] #에러가 발생했을 경우 예외 리턴
