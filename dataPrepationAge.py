

#mathématiques et le stockage efficace de données numériques.
import numpy as np
#lecture et traitement des images
import cv2
#manipulation de fichiers et de répertoires
import os
#lecture de fichiers CSV
import pandas as pd
#TensorFlow pour la gestion des données et le traitement des images
import tensorflow as tf
#to_categorical pour la conversion des étiquettes en format catégoriel
from tensorflow.keras.utils import to_categorical
#train_test_split pour diviser les dosssnnées en ensembles d'entraînement et de testfrom sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split



# Store images and ages 
# number of age categories
pixels = []
agesLabel = []
num_classes = 7  

path = ["UTKFace/", "combined_faces/"]



def img_shaping(path,img,i):
    
    #reas the img from the directory
    #read the img as array
    img=cv2.imread(str(path)+str(img))
    if img is None:
        print(f"[SKIP] Could not read: {img}")
        return None
    
    #color gray for norgb and 1 dimenson
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #resize the img
    img=cv2.resize(img,(200,200))
    print("reading the imgs......."+str(i)+"/"+str(len(os.listdir(path))))
    
    return img



# Function to map age to class labels
def class_labels_reassign(age):
    if 1 <= age <= 2:
        return 0
    elif 3 <= age <= 9:
        return 1
    elif 10 <= age <= 20:
        return 2
    elif 21 <= age <= 27:
        return 3
    elif 28 <= age <= 45:
        return 4
    elif 46 <= age <= 65:
        return 5
    else:
        return 6



def dataPreparation(path,aug=0):
    
    print("reading the imgs.......")

    pixels.clear()
    agesLabel.clear()
    
    i=0
    
    for p in path:
        #os.listdir() all files in the directory
        #img.split("_") → ['25', '1', '2', '20170116174525125.jpg']
        for img in os.listdir(p):    
            i=i+1
            # Skip non-images
            if os.path.isdir(os.path.join(p, img)):
                print(f"[SKIP] Folder: {img}")
                continue

            # Skip non-image files
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"[SKIP] Not an image: {img}")
                continue

            # Check filename format: age_gender_race_date.jpg
            parts = img.split("_")
            if len(parts) < 2:
                print(f"[SKIP] Bad filename format: {img}")
                continue

            # Extract gender safely
            try:
                age = int(parts[0])
            except ValueError:
                print(f"[SKIP] Invalid gender value: {img}")
                continue
            
            #fuction call to reshape the img
            #uniform the size of the imgs
            img = img_shaping(p, img,i)
            if img is None:
                continue
            
            #save the img in the array
            pixels.append(img)        
            
            #gender save
            agesLabel.append(class_labels_reassign(age))
            
            if aug==1 or aug==2:
                img_flipped = cv2.flip(img,0)
                print("reading the imgs......."+str(i)+"/"+str(len(os.listdir(p))))
                pixels.append(img_flipped)
                agesLabel.append(class_labels_reassign(age))
                if aug==2:
                    img_flipped = cv2.flip(img,1)
                    print("reading the imgs......."+str(i)+"/"+str(len(os.listdir(p))))
                    pixels.append(img_flipped)
                    agesLabel.append(class_labels_reassign(age))

    
    
    X = np.array(pixels).reshape(-1, 200, 200, 1).astype('float32') / 255.0
    Y = np.array(agesLabel)
    Y = to_categorical(Y, num_classes)

    print("reading the imgs is compleated successfully")
    print(len(pixels))
    print(len(agesLabel))
    
    return X,Y



dataPreparation(path)



def split_data(pixels,agesLabel,test_size=0.2,random_state=100):
    
    #split data to train and test
    x_train,x_test,y_train,y_test=train_test_split(pixels,agesLabel,random_state=random_state,test_size=test_size)
    
    print("splitting the data is compleated successfully")
    print(str(len(x_train))+"  X_train")
    print(str(len(x_test))+"  X_test")
    print(str(len(y_train))+"  Y_train")
    print(str(len(y_test))+"  Y_test")
    
    return x_train,x_test,y_train,y_test


