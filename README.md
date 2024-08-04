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
2. 다운 받은 파일들을 아래 코드에 넣어 가우시안 블러 데이터 전처리를 진행한다. 이때 빨간색 사진을 위한 전처리 폴더와 초록색 사진을 위한 전처리 폴더 2개로 나누어 사진을 2종류로 나누어 저장한다.(output_folder = "blurred_image" 여기서 폴더명을 바꾸고선 데이터 전처리 진행하는 것을 추천함.)

```python
# 배경 블러링 코드

import cv2
import numpy as np
from google.colab import files
import os
import shutil

# 사용자로부터 이미지를 업로드 받음
uploaded = files.upload()

# 결과 이미지를 저장할 폴더 생성
output_folder = "blurred_image"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 초록색 범위 설정
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# 빨간색 범위 설정 (빨간색은 두 개의 범위로 나뉘어 있음)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# 함수: 가장 큰 연결 요소 찾기
def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    largest_mask = np.zeros_like(mask)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return largest_mask

for image_path in uploaded.keys():
    try:
        # 이미지 읽기
        img = cv2.imdecode(np.frombuffer(uploaded[image_path], dtype=np.uint8), cv2.IMREAD_COLOR)

        # 이미지를 HSV 색 공간으로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 초록색 마스크 생성
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # 빨간색 마스크 생성
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 초록색과 빨간색 마스크 결합
        mask = cv2.bitwise_or(mask_green, mask_red)

        # 가장 큰 마스크 찾기 (초록색 및 빨간색 중)
        largest_mask_green = find_largest_contour(mask_green)
        largest_mask_red = find_largest_contour(mask_red)

        if largest_mask_green is not None and largest_mask_red is not None:
            largest_mask = largest_mask_green if np.sum(largest_mask_green) > np.sum(largest_mask_red) else largest_mask_red
        elif largest_mask_green is not None:
            largest_mask = largest_mask_green
        elif largest_mask_red is not None:
            largest_mask = largest_mask_red
        else:
            largest_mask = None

        if largest_mask is not None:
            # 마스크 확장 (영역을 더 크게 만들기 위해)
            kernel = np.ones((4, 4), np.uint8)
            largest_mask_dilated = cv2.dilate(largest_mask, kernel, iterations=1)

            # 반전된 마스크 생성 (가장 큰 마스킹 영역 이외)
            mask_inv = cv2.bitwise_not(largest_mask_dilated)

            # 원본 이미지에서 가장 큰 마스킹 영역만 추출
            colored_part = cv2.bitwise_and(img, img, mask=largest_mask_dilated)

            # 원본 이미지를 블러 처리
            blurred_img = cv2.GaussianBlur(img, (31,31), 0)

            # 블러 처리된 이미지에서 가장 큰 마스킹 영역 이외의 부분만 추출
            background_part = cv2.bitwise_and(blurred_img, blurred_img, mask=mask_inv)

            # 가장 큰 마스킹 영역과 블러 처리된 배경 결합
            result = cv2.add(colored_part, background_part)

            # 결과 이미지 저장
            output_path = os.path.join(output_folder, f"output_{image_path}")
            cv2.imwrite(output_path, result)
            print(f"Processed image saved as {output_path}")
        else:
            print(f"No significant green or red regions detected in {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# 결과 이미지를 Zip 파일로 압축
shutil.make_archive(output_folder, 'zip', output_folder)

# 압축된 Zip 파일 다운로드
files.download(f"{output_folder}.zip")
```




3. 빨간색 신호등과 초록색 신호등 각각의 전처리된 데이터를 EDGE IMPULSE 사이트에 접속하여 회원가입 한 후, Data Acquisition > ADD DATA를 클릭한다.
![11111111111111111111](https://github.com/user-attachments/assets/5bb83d68-8f99-4858-8f3a-e1eab6fb15ee)


4. training을 선택하고 전처리된 초록색/빨간색 데이터(Original_green_traffic_pictures_from_EV3,Original_red_traffic_pictures_from_EV3를 전처리해서 나온 결과)를 각각 다른 label을 설정한 다음 추가하고 업로드한다. 이후, test 를 선택하고 test 데이터(RED_TEST_PICTURES,GREEN_TEST_PICTURES)를 각각 다른 label로 설정하여 추가한다.
![222222222222222222222222222222222222](https://github.com/user-attachments/assets/0ad68654-f7cb-401d-8967-aba0205eb70c)

5. Create Impulse에 들어가서 Add an input Block 을 클릭.
![image](https://github.com/user-attachments/assets/ded38b00-afc9-4026-9348-b1cc60c26d42)

6. 클릭후 IMAGES에 해당하는 ADD 버튼을 누르면 사진의 수치를 조정할 수 있는 칸이 뜨는데, 계산속도와 불필요한 정보를 제외하고자 48 * 48의 수치로 조정하고 Save Impulse를 클릭하여 변경내용 저장.
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

먼저 데이터 전처리를 진행했을때, 객체를 최소 50% 이상을 탐지한것을 객체 탐지 성공, 그 이하를 객체 탐지 실패라 분류했을때, 
빨간색 신호등 1000장중 154장의 사진이 객체 탐지가 50 % 미만 or 객체 탐지 실패하였음.
초록색 신호등 1000장중 112장의 사진이 객체 탐지가 50 % 미만 or 객체 탐지 실패하였음.
![image](https://github.com/user-attachments/assets/e90ba7b5-e9a7-44a8-89ae-03e03659e4af)


데이터 전처리를 진행하지 않은 데이터를 원본 데이터라 지칭했을때, 사진 전체에 블러링을 적용한 CNN 모델은 성능 향상이 이루어지지 않았음.  
반면, 객체를 제외한 배경만을 블러링한 데이터를 바탕으로 CNN 분류 모델 학습을 진행했을때, 10~25 사이 에포크 구간에서 원본 데이터를 학습시킨 모델의 성능에 비해 성능 향상이 이루어짐.

아래 사진은 데이터 전처리를 거치치 않은 원본 데이터를 학습시킨 모델, 배경 전체에 블러링을 적용한 데이터를 학습시킨 모델, 객체 제외하고 나머지 배경을 블러링한 데이터를 학습시킨 모델 각각을 10번씩 테스트를 진행했을때 나온 정확도와 로스의 평균을 비교한 그래프이다.
![image](https://github.com/user-attachments/assets/f501623b-e25f-4d07-a92c-3e8e8b196184)

![image](https://github.com/user-attachments/assets/51d59257-fa6f-47a0-b577-f834308ae8f7)





# 저자
최은실/eunsil0733@naver.com, 한재웅/hanjaewoong1233@gmail.com

