project.py : dataset.csv를 읽어들인 후 단문별 감정분류를 시행합니다. Bi-LSTM + attention layer가 적용되었습니다.
또 모델 학습 이전 komoran을 통한 형태소 분석기로 토크나이징을 수행합니다.
이후 Project_model.h5 (모델 가중치 파일)와 wordIndex.json (토크나이징을 위한 인덱스 파일) 을 출력하여 predict에 사용합니다.
dataset.csv는 Sentence, Emotion 열로 구성되어, Emotion에는 중립, 행복, 분노, 슬픔, 공포의 다섯가지 감정으로 분류되어 있습니다.

temp.py : 텍스트 전처리를 위해 주로 사용되는 토큰 개수를 셉니다. https://wikidocs.net/44249 를 참고하였습니다.

test.py : predict에 Project_model.h5과 wordIndex.json이 필요합니다. Diary.py와 Model.py를 읽어들여 문장을 입력받고 감정분류 리스트를 출력합니다. \n을 기준으로 슬라이싱합니다.

결과 :  
epoch 3 : train loss 0.661 (acc 0.754), test loss 1.045 (acc 0.618)
epoch 4 : train loss 0.549 (acc 0.796), test loss 1.158 (acc 0.608)

* 현재 로컬 우분투 환경에서 실행시 랜덤하게 exception이 raise되는 버그가 존재합니다만, 타 환경에서는 문제가 발생하지 않아 보류중입니다.
