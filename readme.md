project.py : dataset.csv를 읽어들인 후 단문별 감정분류를 시행합니다. Bi-LSTM + attention layer가 적용되었습니다.
또 모델 학습 이전 okt(https://github.com/open-korean-text/open-korean-text)을 통한 형태소 분석기로 토크나이징을 수행합니다.
이후 Project_model.h5 (모델 가중치 파일)와 wordIndex.json (토크나이징을 위한 인덱스 파일) 을 출력하여 predict에 사용합니다.
dataset.csv는 Sentence, Emotion 열로 구성되어, Emotion에는 중립, 행복, 분노, 슬픔, 공포의 다섯가지 감정으로 분류되어 있습니다.<br>
<br>
내용은 다음과 같습니다.<br>
<br>
Sentence|Emotion<br>
제가 한가지 고민이 생겼습니다.|공포<br>
짝남이 자기 왕이래요......|공포<br>
...<br>
<br>
(7/16)
각종 전처리 기법과 정규화, 부트스트랩을 적용한 결과
26964 train set 기반 530 test set에 대해 다음과 같은 성능을 나타냈습니다.

1. vanilla model (24000 test, 2964 validation) 62.64%
2. 형태소 정규화 (okt.morphs() 메소드의 norm, stem) 65.85%
3. 10 bootstrap 모델 68.11%
4. 각종 특수기호 정규화, 숫자 정규화 67.74%
5. 토큰수 제한 (빈도수 thres>2) 69.06%

test.py : predict에 Project_model.h5과 wordIndex.json이 필요합니다. Diary.py와 Model.py를 읽어들여 문장을 입력받고 감정분류 리스트를 출력합니다. 문장 분리 알고리즘은 다음 패키지를 사용하였습니다. http://docs.likejazz.com/kss/

