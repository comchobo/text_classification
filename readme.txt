project.py : dataset.csv를 읽어들인 후 단문별 감정분류를 시행합니다. Bi-LSTM + attention layer가 적용되었습니다.
이후 Project_model.h5와 wordIndex.json을 출력하여 predict에 사용합니다.
temp.py : 텍스트 전처리를 위해 주로 사용되는 토큰 개수를 셉니다. https://wikidocs.net/44249를 참고하였습니다.

test.py : Diary.py와 Model.py를 읽어들여 문장을 입력받고 감정분류 리스트를 출력합니다. \n을 기준으로 슬라이싱합니다.
