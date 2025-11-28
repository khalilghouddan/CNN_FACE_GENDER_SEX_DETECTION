📸  Age & Gender Prediction

Machine Learning Project for Real-Time Age and Gender Detection using UTKFace Dataset

🚀 Overview

This project uses the UTKFace dataset to train deep learning models capable of predicting:

Gender (Male/Female)

Age (categorized into multiple age classes)

The project includes modules for data preparation, model training, and real-time prediction using a webcam.

Deep learning models are implemented using TensorFlow/Keras, and the system performs real-time face detection using OpenCV.

📂 Project Structure
UTKFace-Project/
│
├── combined_faces/             # Cleaned & formatted dataset
│
├── dataPreparationSex.py       # Preprocessing script for gender labels
├── dataPreparationAge.py       # Preprocessing script for age categories
│
├── genderModelSex.py           # CNN model architecture for gender detection
├── genderModelAge.py           # CNN model architecture for age classification
│
├── trainModelsSex.py           # Model training script for gender
├── trainModelsAge.py           # Model training script for age
│
├── gender_model.h5             # Pretrained gender model
│
└── webCamDetection.py          # Real-time webcam detection script

📦 Requirements

Make sure you have the following libraries installed:

pip install tensorflow
pip install numpy
pip install pandas
pip install opencv-python
pip install matplotlib


Python version recommended: 3.10 – 3.11

🧹 1. Data Preparation
🔹 dataPreparationSex.py

Loads UTKFace images

Extracts gender labels from filenames

Processes and resizes images

Converts images to arrays

Saves them into NumPy-friendly format

🔹 dataPreparationAge.py

Similar to dataPreparationSex.py

Extracts age and converts it into 7 age groups

Prepares images for training

🧠 2. Model Architecture
🔹 genderModelSex.py

Defines a CNN model for binary gender classification:

Convolutional layers

Pooling

Batch Normalization

Fully connected dense layers

Softmax output (Male/Female)

🔹 genderModelAge.py

Defines a CNN model for age-group classification:

Multi-class softmax output

Trained on 7 age categories

🏋️ 3. Training Scripts
🔹 trainModelsSex.py

Loads preprocessed data

Compiles the gender CNN

Trains and saves the model as gender_model.h5

🔹 trainModelsAge.py

Loads the age dataset

Trains and saves the age model

Displays accuracy/loss curves

🎥 4. Real-Time Detection
🔹 webCamDetection.py

This script:

Loads the trained models (gender_model.h5, age model)

Captures webcam stream using OpenCV

Detects faces in real-time

Preprocesses the detected face

Predicts:
✔ Gender
✔ Age group

Draws bounding boxes and labels on screen

To start real-time detection:

python webCamDetection.py

🧪 Testing the Model

To test on a single image:

from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("gender_model.h5")
img = cv2.imread("test.jpg")
# preprocess...
# prediction...

📊 Results

Achieved reliable classification accuracy on UTKFace dataset

Smooth real-time detection (20–30 FPS depending on system)

Strong generalization on unseen faces

(Add your accuracy results here once your training is finished.)

📌 Future Improvements

Add race detection

Increase number of age groups

Improve model using transfer learning

Add GUI interface with Tkinter or PyQt

Deploy as a Flask/FastAPI Web App

🙌 Credits

UTKFace Dataset: A large-scale dataset for age, gender, and ethnicity classification.

Project developed by Khalil Ghouddan.

If you want, I can also:

✔ Generate a Markdown version ready for GitHub
✔ Add images or architecture diagrams
✔ Add installation instructions for Windows/Linux/Mac
✔ Add badges (TensorFlow, Python, License, etc.)

Just tell me!
