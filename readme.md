project.py : dataset.csv를 읽어들인 후 단문별 감정분류를 시행합니다. Bi-LSTM + attention layer가 적용되었습니다.
이후 Project_model.h5와 wordIndex.json을 출력하여 predict에 사용합니다.
dataset.csv는 sentence, emotion 열로 구성되었습니다.

temp.py : 텍스트 전처리를 위해 주로 사용되는 토큰 개수를 셉니다. https://wikidocs.net/44249 를 참고하였습니다.

test.py : Diary.py와 Model.py를 읽어들여 문장을 입력받고 감정분류 리스트를 출력합니다. \n을 기준으로 슬라이싱합니다.

결과 :  
epoch 3 : train loss 0.661 (acc 0.754), test loss 1.045 (acc 0.618)
epoch 4 : train loss 0.549 (acc 0.796), test loss 1.158 (acc 0.608)

* 현재 로컬 우분투 환경에서 실행시 랜덤하게 exception이 raise되는 버그가 존재합니다만, 타 환경에서는 문제가 발생하지 않아 보류중입니다.
