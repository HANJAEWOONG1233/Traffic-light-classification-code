# 가우시안 블러 전처리 기법의 적용범위에 따른 CNN 분류 모델 성능 향상 연구


(데이터 수집부터 데이터 전처리 과정 CNN 모델 학습 및 결과 나오는거까지 다이어 그램으로 표현)

CNN 이미지 분류 모델을 학습시켜 EV3 이동로봇시스템에 들어가는 신호등을 구분하는 시험을 진행함.
CNN 모델 학습시에 들어가는 가우시안 블러의 데이터 전처리 적용범위을 사진 전체와 객체를 제외한 배경 두 가지로 나누어 성능을 측정함.
데이터 전처리를 진행하지 않은 데이터를 원본 데이터라 지칭했을때, 사진 전체에 블러링을 적용한 CNN 모델의 성능은 유의미한 성능 향상이 이루어지지 않았음.  
반면, 객체를 제외한 배경만을 블러링한 데이터를 바탕으로 CNN 분류 모델 학습을 진행했을때, 동일 수의 데이터 전처리를 진행하지 않은 10% 수준의 성능 향상이 이루어짐.


# 필수 조건
-  Python 3.3+
-  OPENCV
-  EDGE IMPULSE


# 용법
