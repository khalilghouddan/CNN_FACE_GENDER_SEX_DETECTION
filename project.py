





#Python library for numerical computing.
import numpy as np
#Reading and processing images
#OpenCV reads images in BGR format (Blue, Green, Red)
import cv2
#work with file paths and directories
import os
#File pattern matching
import glob
#Converts a PIL Image instance to a Numpy array.
from tensorflow.keras.preprocessing.image import img_to_array
#For random shuffling
#toget a random order of the images for training 
import random


data=[]
labels=[]

#if not os.path.isdir(f) part: Makes sure we only get files, not folders.
#So together, /**/* means: “Go into this folder and all subfolders, and match all files or directories.”
image_files = [f for f in glob.glob(r'C:\emsi\machine learning fundamentals\THEPROJECTAI\genders' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

print(f"📷 Found {len(image_files)} images...")


for img in image_files:

    image = cv2.imread(img)# Read the image
    image = cv2.resize(image, (96,96))# Resize to 96x96
    image = img_to_array(image)# Convert to NumPy array
    
    data.append(image)#save image data to data list
    
    label = img.split(os.path.sep)[-2] # C:\...\THEPROJECTAI\genders\woman\face_1162.jpg
    if label == "woman":
        label = 1
    else:
        label = 0
    labels.append([label]) # [[1], [0], [0], ...] save label to labels list
    

#image.shape = (96, 96, 3)
#[[[255, 128, 0], [255, 130, 5], ...], ...]
# Normalize pixel values to the range [0, 1]
#image[0,0] = [1.0, 0.502, 0.0]
data = np.array(data, dtype="float") / 255.0

#labels = [[1], [0], [1], [1], [0], ...]
labels = np.array(labels)   


# print("Data shape:", data.shape)
# print("Labels shape:", labels.shape)
# Data shape: (2307, 96, 96, 3)
# Labels shape: (2307, 1)


from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

def to_categorical(y, num_classes):
    """Convert class vector (integers) to binary class matrix."""
    return np.eye(num_classes)[y.reshape(-1)]
# Convert labels to categorical (one-hot encoding)
#binary classification  [[1,0], [0,1], ... ]

trainY = to_categorical(trainY, num_classes=2)
testY  = to_categorical(testY, num_classes=2)





from tensorflow.keras.preprocessing.image import ImageDataGenerator

#data augmentation to prevent overfitting
#create an instance of the ImageDataGenerator class with various augmentation parameters
#from older img fliping and rotation techniques to more advanced methods like zooming and shearing
aug = image_data_generator = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")


#Build the CNN model
#Sequential model: layers are added one after another in a linear stack.
#Sequential is suitable for simple models where each layer has exactly one input tensor and one output tensor.
from tensorflow.keras.models import Sequential

#Conv2D layer: applies convolution operation to the input data.
from tensorflow.keras.layers import Conv2D

#MaxPooling2D layer: reduces the spatial dimensions (width and height) of the input volume.
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense   
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K




#fonction of the model
def build_model(width, height, depth, classes):
    
    model = Sequential()
    inputShape = (width, height, depth)
    
    #channel dimension
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = ( height, width,depth )
        chanDim = 1
        
    
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation("sigmoid"))
    try:
        plot_model(model, to_file="model_schema.png", show_shapes=True, show_layer_names=True)
        print("\n✅ Schema saved as 'model_schema.png'")
    except Exception as e:
        print("\n⚠️ Could not generate image schema:", e)
    
    return model



# Build the model
model = build_model(96,96,1,2)
# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


print("\n🚀 Training model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY),
    epochs=10,
    verbose=1
)





# Save the model to disk
model.save("gender_detection_model.h5")
print("\n✅ Model saved to disk as 'gender_detection_model.h5'")


from matplotlib import pyplot as plt

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 10), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 10), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 10), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 10), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.savefig("training_plot.png")


