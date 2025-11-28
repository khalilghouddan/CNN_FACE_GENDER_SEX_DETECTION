# 📸 UTKFace Age & Gender Detection  
Deep Learning Project for Real-Time Face Analysis

This project uses the **UTKFace dataset** to train Convolutional Neural Networks capable of predicting **gender** and **age groups** from facial images.  
It includes data preparation scripts, model architectures, training pipelines, and a real-time webcam detection system powered by OpenCV.

---

## 📂 Project Structure

UTKFace-Project/
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

yaml
Copy code

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

Extracts gender labels from filenames

Resizes and normalizes images

Saves arrays for training gender model

▶ dataPreparationAge.py
Extracts age values and converts them into 7 age groups

Preprocesses images

Prepares dataset for age-classification training

🧠 Model Architectures
▶ genderModelSex.py
Defines the CNN for binary gender classification:

Convolution + Pooling layers

Batch Normalization

Dense layers

Softmax output (Male / Female)

▶ genderModelAge.py
Defines the CNN for multi-class age classification:

Softmax output over 7 age categories

Deeper CNN structure for better feature extraction

🏋️ Training the Models
▶ trainModelsSex.py
Loads gender dataset

Trains CNN model

Saves result as gender_model.h5

▶ trainModelsAge.py
Loads the age dataset

Trains age classification model

Displays accuracy/loss curves

🎥 Real-Time Webcam Detection
▶ webCamDetection.py
This script performs:

Face detection using OpenCV

Image preprocessing

Real-time gender prediction

Real-time age group prediction

Drawing bounding boxes + labels on the webcam feed

Run the script:

bash
Copy code
python webCamDetection.py
📊 Results
Works in real-time (20–30 FPS depending on hardware)

Good performance on UTKFace for gender classification

Age classification accuracy depends on dataset quality

(Add your own accuracy metrics here.)

🚀 Future Improvements
Add race/ethnicity classification

Improve age detection using transfer learning (ResNet, MobileNet, etc.)

Build a user interface (Tkinter / PyQt)

Deploy as a web application using Flask or FastAPI

🙌 Credits
UTKFace Dataset — A benchmark dataset for age, gender, and ethnicity detection

Developed by Khalil Ghouddan

If you want, I can also add:
✔ Badges (Python, TensorFlow, License, Stars, etc.)
✔ Screenshots or GIFs of real-time detection
✔ A license section (MIT / Apache / GPL)
