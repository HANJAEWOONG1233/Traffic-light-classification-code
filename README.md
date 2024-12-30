#  Enhancing CNN Classification Model Performance Through Gaussian Blur Preprocessing Techniques 

## 🚀 Introduction

### 🎯 Objective of the Study
Train a CNN classification model with the default settings in **EDGE IMPULSE** to compare the performance of traffic light classification for **EV3 Robots**. Below is the CNN architecture used:
![CNN Architecture](https://github.com/user-attachments/assets/53e2f781-e3b5-491a-987f-c67b4715ebfe)

The study measures performance by applying Gaussian blur preprocessing to the dataset in two scenarios:
1. Blurring the entire image.
2. Blurring only the background, excluding the traffic light objects.

---

## 📋 Prerequisites
- **Python 3.8+**
- **OpenCV**
- **EDGE IMPULSE**

---

## 🛠️ Methodology

1. **Download Original Files**:
   - Obtain the original files (`Original_green`, `Original_red`) from the repository.

2. **Data Preprocessing**:
   - Place the downloaded files into the `Data_preprocessing_code` directory.
   - Perform Gaussian blur preprocessing for:
     - Entire image blurring for red traffic light images.
     - Entire image blurring for green traffic light images.
     - Background blurring excluding red traffic light objects.
     - Background blurring excluding green traffic light objects.
   - **Recommendation**: Change the folder name from `output_folder = "blurred_image"` before proceeding with data preprocessing.
   
   - **Background Blurring Code & Flowchart**:
     ![Preprocessing Flowchart](https://github.com/user-attachments/assets/7c9e306b-c45c-419f-873a-d09f6b0b63c9)

   - **Reason for HSV Conversion**:
     - HSV conversion allows for better recognition of green and red colors based on hue, saturation, and value, compared to RGB.
     - Apply masking to detect traffic lights using color thresholds.
     - **Limitation**: This method may mistakenly identify other objects as traffic lights, leading to unintended background blurring.

3. **Data Upload to EDGE IMPULSE**:
   - Sign up on the **EDGE IMPULSE** website.
   - Navigate to **Data Acquisition > ADD DATA**.
   ![Add Data](https://github.com/user-attachments/assets/5bb83d68-8f99-4858-8f3a-e1eab6fb15ee)

4. **Labeling and Uploading Preprocessed Data**:
   - Select **training** and upload the preprocessed green and red traffic light images (`Original_green_traffic_pictures_from_EV3`, `Original_red_traffic_pictures_from_EV3`) with distinct labels.
   - Select **test** and upload the test datasets (`RED_TEST_PICTURES`, `GREEN_TEST_PICTURES`) with appropriate labels.
   ![Labeling Data](https://github.com/user-attachments/assets/0ad68654-f7cb-401d-8967-aba0205eb70c)

5. **Create Impulse**:
   - Go to **Create Impulse** and click on **Add an input Block**.
   ![Add Input Block](https://github.com/user-attachments/assets/ded38b00-afc9-4026-9348-b1cc60c26d42)

6. **Configure Image Settings**:
   - Click the **ADD** button under **IMAGES**.
   - Adjust image dimensions to **48x48** to ensure consistent data and improve computation speed.
   - **Reason**: Maintaining uniform image size prevents data bias and overfitting.
   - Save the impulse configuration.
   ![Configure Images](https://github.com/user-attachments/assets/33ef9e2b-11f4-4728-8dfd-3ed34941a65e)

7. **Generate Features**:
   - Navigate to **IMAGE > GENERATE FEATURES > GENERATE FEATURES** to create the model.
   ![Generate Features](https://github.com/user-attachments/assets/51f7b51a-cbdb-4331-99a0-406953ef06b8)

8. **Train the Classifier**:
   - Go to **Classifier**.
   - Set **Epochs** to `10` and **Learning Rate** to `0.0005`.
   - Click **Start Training** to begin training the CNN classification model using the preprocessed data.
   ![Start Training](https://github.com/user-attachments/assets/b530f487-a4d1-4afa-917d-0b2d96c2f49e)

9. **Evaluate Model Performance**:
   - Check **Accuracy** and **Loss** on the validation data.
   - Click **Model Testing** and then **Classify All** to perform classification using the trained model.
   ![Evaluate Model](https://github.com/user-attachments/assets/cc80b0ee-e342-40d3-ab64-ac113cf07325)
   ![Classify All](https://github.com/user-attachments/assets/8ed19cca-66ec-4dcf-b9f2-efbca8b19cb3)

10. **Test the Model**:
    - Measure the accuracy and record the loss using the test dataset.
    ![Test Accuracy](https://github.com/user-attachments/assets/14946022-1faf-4652-9fee-9926d17a4ecc)

---

## 🖼️ Preprocessing Results

- **Original Red Traffic Light Images**:
  ![Original Red](https://github.com/user-attachments/assets/e0a58f3b-479f-4202-a30c-c7b79d9159f8)

- **Original Green Traffic Light Images**:
  ![Original Green](https://github.com/user-attachments/assets/6f026c49-5648-4c08-9293-f99323b7cc1c)

- **Red Traffic Light Images with Entire Image Blurred**:
  ![Blurred Red Entire](https://github.com/user-attachments/assets/a9677f3d-ab1d-465d-8845-0b2bfbb1243d)

- **Green Traffic Light Images with Entire Image Blurred**:
  ![Blurred Green Entire](https://github.com/user-attachments/assets/babbc883-b026-4c44-a1b0-54b6b07b34a6)

- **Red Traffic Light Images with Background Blurred**:
  ![Blurred Red Background](https://github.com/user-attachments/assets/496b9c63-7d3c-4537-85df-35b7d413c3f8)

- **Green Traffic Light Images with Background Blurred**:
  ![Blurred Green Background](https://github.com/user-attachments/assets/3db2574f-16b7-4e9a-b9ae-2303e10ba05c)

- **Red Traffic Light Images with Failed Object Detection**:
  ![Red Detection Failure](https://github.com/user-attachments/assets/ac341265-94c3-4002-8d04-ad8509f59f70)

- **Green Traffic Light Images with Failed Object Detection**:
  ![Green Detection Failure](https://github.com/user-attachments/assets/5204ce20-6d21-477f-b19b-b6cc6a5b73f8)

---

## 📊 Performance Comparison Results

### 🏅 Object Detection Success vs. Failure
- **Criteria**: Successful object detection if at least 50% of the object is detected.
- **Results**:
  - **Red Traffic Lights**: 154 out of 1000 images failed object detection.
  - **Green Traffic Lights**: 112 out of 1000 images failed object detection.

**Future Improvement**: Increasing the success rate of object detection in background-blurred preprocessing is expected to enhance overall performance.

![Detection Success](https://github.com/user-attachments/assets/b2f90ee8-d900-4597-aad8-32c95c5dd31d)

### 📈 Model Performance Without Preprocessing
- **Entire Image Blurring**: No performance improvement observed.
- **Background Blurring**:
  - **Performance Boost**: Achieved higher accuracy compared to models trained on original data between epochs 10-25.

### 📉 Accuracy and Loss Comparison
The graph below compares the average accuracy and loss at epochs 10, 15, 20, and 25 for three models:
1. Trained on original data without preprocessing.
2. Trained on entire image blurred data.
3. Trained on background-blurred data.

![Performance Graph](https://github.com/user-attachments/assets/033890af-be2f-4b03-9dc3-8a568f804346)

### 📊 Detailed Metrics
Analyzing the graph's metrics and summarizing them in a table:

<img width="495" alt="image" src="https://github.com/user-attachments/assets/75bb47f8-aeab-4c15-a67c-01fb1568477c" />


**Insight**: Even with a 10% reduction in the number of images, using background-blurred preprocessing achieves equivalent or superior performance.

---

## 👥 Authors
- **Eunsil Choi** - eunsil0733@naver.com
- **JaeWoong Han** - hanjaewoong1233@gmail.com

