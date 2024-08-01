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
1. 원본 파일들을 다운로드 하세요.
2. 다운 받은 파일들을 아래 링크코드를 활용하여 데이터 전처리를 진행합니다. 이때 빨간색 사진을 위한 전처리 폴더와 초록색 사진을 위한 전처리 폴더 2개로 나누어 사진을 2종류로 나누어 저장한다.(output_folder = "blurred_image" 여기서 폴더명을 바꾸고선 데이터 전처리 진행하는 것을 추천함.)

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

3. 빨간색 신호등과 초록색 신호등 각각의 전처리된 데이터를 EDGE IMPULSE 사이트에 접속하여 회원가입 한 후, 데이터를 추가한다.





