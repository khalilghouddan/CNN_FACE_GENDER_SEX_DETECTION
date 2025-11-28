# 📸  Age & Gender Detection  
Deep Learning Project for Real-Time Face Analysis

This project uses the **UTKFace dataset** to train Convolutional Neural Networks capable of predicting **gender** and **age groups** from facial images.  
It includes data preparation scripts, model architectures, training pipelines, and a real-time webcam detection system powered by OpenCV.

---

## 📂 Project Structure

```bash
Project/
│
├── combined_faces/ # Cleaned & formatted dataset
│
├── dataPreparationSex.py # Preprocessing script for gender labels
├── dataPreparationAge.py # Preprocessing script for age categories
│
├── genderModelSex.py # CNN model architecture for gender detection
├── genderModelAge.py # CNN model architecture for age classification
│
├── trainModelsSex.py # Model training script for gender
├── trainModelsAge.py # Model training script for age
│
├── gender_model.h5 # Pretrained gender model
│
└── webCamDetection.py # Real-time webcam detection script
```


---

## ⚙️ Installation

Make sure you have Python **3.10 – 3.11** installed.

Install dependencies:

```bash
pip install tensorflow
pip install numpy
pip install pandas
pip install opencv-python
pip install matplotlib
🧹 Data Preparation
▶ dataPreparationSex.py
Loads UTKFace images
```


📊 Results
Works in real-time (20–30 FPS depending on hardware)

<p align="center">
  <img src="./imgs_project/age_model_result.png" width="400">
  <img src="./imgs_project/sex_model_result.png" width="400">
</p>

## 📊 Results

### Result 1
![Result Image 1](./images/result1.png)

### Result 2
![Result Image 2](./images/result2.png)



🚀 Future Improvements
Add race/ethnicity classification

Improve age detection using transfer learning (ResNet, MobileNet, etc.)

Build a user interface (Tkinter / PyQt)

Deploy as a web application using Flask or FastAPI

🙌 Credits
UTKFace Dataset 

Developed by Khalil Ghouddan

If you want, I can also add:
✔ Badges (Python, TensorFlow, License, Stars, etc.)
✔ Screenshots or GIFs of real-time detection
✔ A license section (MIT / Apache / GPL)
