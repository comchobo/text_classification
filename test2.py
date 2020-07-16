import Diary
import pandas as pd
k=0
test_data = pd.read_csv("testset0.csv", encoding='UTF8',sep='|' )
for i in range(len(test_data['s'])):
    sentence = test_data['s'].iloc[i]
    emotion = test_data['d'].iloc[i]
    response = Diary.predictTEST(sentence)
    if response != emotion:
        print(sentence," 해당 문장은 ", response, "로 분류되었으나", emotion, "가 실제 감정입니다.\n")
        k=k+1

print("정답율은 ", 1- k/len(test_data), "입니다")

