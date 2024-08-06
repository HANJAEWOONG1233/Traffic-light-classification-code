# 가우시안 블러 전처리 기법의 적용범위에 따른 CNN 분류 모델 성능 향상 연구

EDGE IMPULSE에 기본 세팅값의 CNN 분류 모델을 학습시켜, EV3 로봇에 들어가는 신호등을 구분하는 성능 비교를 진행합니다. 아래는 사용된 CNN 구조입니다.
![image](https://github.com/user-attachments/assets/53e2f781-e3b5-491a-987f-c67b4715ebfe)


CNN 모델 학습에 사용되는 가우시안 블러가 사용된 데이터의 적용범위을 사진 전체와 객체를 제외한 배경 두 가지의 사례로 나누어 성능을 측정합니다.

# 필수 조건
-  Python 3.8+
-  OPENCV
-  EDGE IMPULSE


# 방법
1. 레포지토리에 있는 원본 파일(Original_green_traffic_pictures_from_EV3,Original_red_traffic_pictures_from_EV3)을 다운로드 받습니다.

2. 다운 받은 파일들을 Data_preprocessing_code 파일에 있는 코드들에 넣어 사진 전체/객체 제외 배경에 대한 가우시안 블러 데이터 전처리를 진행합니다. 이때 사진 전체를 블러링하는 빨간색 신호등 사진을 위한 폴더, 사진 전체를 블러링하는 초록색 신호등 사진을 위한 폴더, 빨간 신호등 객체 제외 나머지 배경만을 블러링하는 사진들을 위한 폴더, 초록 신호등 객체 제외 나머지 배경만을 블러링하는 사진들을 위한 폴더 4개로 나누어 사진을 4종류로 나누어 저장한다.(output_folder = "blurred_image" 여기서 폴더명을 바꾸고선 데이터 전처리 진행하는 것을 추천함.)


- 사진 전체 블러링 코드 순서도


  ![image](https://github.com/user-attachments/assets/df42ae63-efc5-450d-862c-e2fb47f19489)




- 객체 제외 나머지 배경 블러링 코드 순서도


    ![image](https://github.com/user-attachments/assets/7c9e306b-c45c-419f-873a-d09f6b0b63c9)




    HSV변환을 적용하는 이유는 색상,명도,채도를 바탕으로 기존 RGB보다 더 다양한 영역의 초록색,빨간색을 인식할 수 있기 때문에, HSV 변환하여 마스킹을 적용.
    색상 마스킹을 활용하여 신호등을 찾기 위해, 사진내에서 찾은 마스킹들중 제일 큰 것을 신호등으로 인식하게 함.
    (다만 이 방법은 사진에 따라 다른 물체를 신호등으로 인식해서 나머지를 블러처리 한다는 한계점이 존재.)




3. 빨간색 신호등과 초록색 신호등 각각의 전처리된 데이터를 EDGE IMPULSE 사이트에 접속하여 회원가입 한 후, Data Acquisition > ADD DATA를 클릭한다.
![11111111111111111111](https://github.com/user-attachments/assets/5bb83d68-8f99-4858-8f3a-e1eab6fb15ee)



4. training을 선택하고 전처리된 초록색/빨간색 데이터(Original_green_traffic_pictures_from_EV3,Original_red_traffic_pictures_from_EV3를 전처리해서 나온 결과)를 각각 다른 label을 설정한 다음 추가하고 업로드한다. 이후, test 를 선택하고 test 데이터(RED_TEST_PICTURES,GREEN_TEST_PICTURES)를 각각 다른 label로 설정하여 추가한다.
![222222222222222222222222222222222222](https://github.com/user-attachments/assets/0ad68654-f7cb-401d-8967-aba0205eb70c)


5. Create Impulse에 들어가서 Add an input Block 을 클릭.
![image](https://github.com/user-attachments/assets/ded38b00-afc9-4026-9348-b1cc60c26d42)


6. 클릭후 IMAGES에 해당하는 ADD 버튼을 누르면 사진의 수치를 조정할 수 있는 칸이 뜨는데, 계산속도와 일정한 사진 크기를 위해 48 * 48의 수치로 조정하고 Save Impulse를 클릭하여 변경내용 저장. (사진을 일정한 크기로 조정하는 이유는 데이터간의 일관성을 확보하고, 성능을 개선하기 위함임. 데이터 일관성 미확보시 특정 데이터 셋에 의한 데이터 편향이나 데이터 과적합 발생할 수 있음.)
![66666666666666666](https://github.com/user-attachments/assets/33ef9e2b-11f4-4728-8dfd-3ed34941a65e)


7. IMAGE > GENERATE FEATURES > GENERATE FEATURES 를 클릭해서, 모델을 생성.
![8888888888888888888888](https://github.com/user-attachments/assets/51f7b51a-cbdb-4331-99a0-406953ef06b8)


8. Classifier 에 들어가서 에포크(10), 학습률(0.0005)로 설정하여 start training을 눌러 전처리된 데이터를 바탕으로 CNN 분류 모델 훈련을 시작.(CNN 모델은 EDGE IMPULSE에서 기본적으로 주어지는 간단한 CNN 모델을 사용.)
![333333333333333333333333333](https://github.com/user-attachments/assets/b530f487-a4d1-4afa-917d-0b2d96c2f49e)


9. valid 데이터에 대한 accuracy와 loss를 확인하고, model testing을 클릭한후, classify all을 클릭하여 학습한 모델을 바탕으로 분류를 진행한다.

![image](https://github.com/user-attachments/assets/cc80b0ee-e342-40d3-ab64-ac113cf07325)
![44444444444444444444444444444444](https://github.com/user-attachments/assets/8ed19cca-66ec-4dcf-b9f2-efbca8b19cb3)


10. test 데이터를 바탕으로 모델 테스트를 진행했을때의 정확도를 측정하고, 6번에서 구했던 로스를 기록.
![image](https://github.com/user-attachments/assets/14946022-1faf-4652-9fee-9926d17a4ecc)


# 전처리 결과

- EV3 빨간 신호등 사진 원본 데이터
![result_image_grid (7)](https://github.com/user-attachments/assets/e0a58f3b-479f-4202-a30c-c7b79d9159f8)


- EV3 초록 신호등 사진 원본 데이터
![result_image_grid (2)](https://github.com/user-attachments/assets/6f026c49-5648-4c08-9293-f99323b7cc1c)


- EV3 빨간 신호등 사진 전체 블러링 데이터
![result_image_grid (11)](https://github.com/user-attachments/assets/a9677f3d-ab1d-465d-8845-0b2bfbb1243d)


- EV3 초록 신호등 사진 전체 블러링 데이터
![result_image_grid (10)](https://github.com/user-attachments/assets/babbc883-b026-4c44-a1b0-54b6b07b34a6)


- EV3 빨간 신호등 사진 객체 제외 블러링 데이터
![result_image_grid (8)](https://github.com/user-attachments/assets/496b9c63-7d3c-4537-85df-35b7d413c3f8)


- EV3 초록 신호등 사진 객체 제외 블러링 데이터
![result_image_grid (9)](https://github.com/user-attachments/assets/3db2574f-16b7-4e9a-b9ae-2303e10ba05c)


- EV3 빨간 신호등 사진 객체 탐지 실패 데이터 사례
  ![result_image_grid (12)](https://github.com/user-attachments/assets/ac341265-94c3-4002-8d04-ad8509f59f70)


- EV3 초록 신호등 사진 객체 탐지 실패 데이터 사례
  ![result_image_grid (13)](https://github.com/user-attachments/assets/5204ce20-6d21-477f-b19b-b6cc6a5b73f8)




# 성능 비교 결과

먼저 데이터 전처리를 진행했을때, 객체를 최소 50% 이상을 탐지한것을 객체 탐지 성공, 그 이하를 객체 탐지 실패라 분류했을때, 빨간색 신호등 1000장중 154장의 사진, 초록색 신호등 1000장중 112장의 사진들이 객체 탐지가 실패하였음.
앞으로 객체 제외 배경 블러링 데이터 전처리의 객체 탐지 성공 확률을 올리게 된다면, 성능이 더 향상될 것으로 예상함.

<img width="467" alt="image" src="https://github.com/user-attachments/assets/b2f90ee8-d900-4597-aad8-32c95c5dd31d">







데이터 전처리를 진행하지 않은 데이터를 원본 데이터라 지칭했을때, 사진 전체에 블러링을 적용한 CNN 모델은 성능 향상이 이루어지지 않았음.  
반면, 객체를 제외한 배경만을 블러링한 데이터를 바탕으로 CNN 분류 모델 학습을 진행했을때, 10~25 사이 에포크 구간에서 원본 데이터를 학습시킨 모델의 성능에 비해 성능 향상이 이루어짐.


아래 사진은 데이터 전처리를 거치치 않은 원본 데이터를 학습시킨 모델, 배경 전체에 블러링을 적용한 데이터를 학습시킨 모델, 객체 제외하고 나머지 배경을 블러링한 데이터를 학습시킨 모델 3가지
각각을 에포크 10,15,20,25에서 각각 10번씩 테스트를 진행했을때 나온 정확도와 로스의 평균을 비교한 그래프이다.


<img width="805" alt="image" src="https://github.com/user-attachments/assets/033890af-be2f-4b03-9dc3-8a568f804346">







![image](https://github.com/user-attachments/assets/d76b3734-b501-4778-ae48-b2dee5d15ede)





 위 그래프의 수치를 분석해서 표로 정리하면,

   <img width="545" alt="image" src="https://github.com/user-attachments/assets/4d8428fa-be45-4559-b0ae-59e57de25edb">


   약 10%의 사진 갯수가 감소하더라도, 객체 제외 배경 블러링을 활용한다면 동등 이상의 효과를 가짐.






# 저자
최은실/eunsil0733@naver.com, 한재웅/hanjaewoong1233@gmail.com

