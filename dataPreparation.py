

#clacule math
import numpy as np

#reading img detection faces drawing ....
import  cv2


#os library for file and directory manipulation
import os

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical




#stor the imgs
pixels=[]

#stor ages
age=[]

#stor genders
gender=[]

def img_shaping(path,img,i):
    
    #reas the img from the directory
    #read the img as array
    img=cv2.imread(str(path)+str(img))
    
    #color gray for norgb and 1 dimenson
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #resize the img
    img=cv2.resize(img,(200,200))
    print("reading the imgs......."+str(i)+"/"+str(len(os.listdir(path))))

    
    return img


def dataPreparation(path1,aug=0):
    
    print("reading the imgs.......")

    pixels.clear()
    gender.clear()
    
    i=0
    #os.listdir() all files in the directory
    for img in os.listdir(path1):    
        i=i+1
        #img.split("_") → ['25', '1', '2', '20170116174525125.jpg']
        genders=int(img.split("_")[1])
        
        #fuction call to reshape the img
        #uniform the size of the imgs
        img = img_shaping(path1, img,i)
        
        #save the img in the array
        pixels.append(img)        
        
        #gender save
        gender.append(genders)
        
        # if aug==1 or aug==2:
        #     img_flipped = cv2.flip(img,0)
        #     pixels.append(img_flipped)
        #     gender.append(genders)
        #     if aug==2:
        #         img_flipped = cv2.flip(img,1)
        #         pixels.append(img_flipped)
        #         gender.append(genders)

        
        #age save
        #ages=img.split("_")[0]
    
    X = np.array(pixels).reshape(-1, 200, 200, 1).astype('float32') / 255.0
    Y = np.array(gender)
    Y = to_categorical(Y, 2)


    print("reading the imgs is compleated successfully")
    print(len(pixels))
    print(len(gender))
    
    
    return X,Y
    
#dataPreparation(path)

    
def split_data(pixels,gender,test_size=0.2,random_state=100):
    
    #split data to train and test
    x_train,x_test,y_train,y_test=train_test_split(pixels,gender,random_state=random_state,test_size=test_size)
    
    print("splitting the data is compleated successfully")
    print(str(len(x_train))+"  X_train")
    print(str(len(x_test))+"  X_test")
    print(str(len(y_train))+"  Y_train")
    print(str(len(y_test))+"  Y_test")
    
    return x_train,x_test,y_train,y_test




path="UTKFace/"


c,d=dataPreparation(path,2)
f=split_data(c,d)





















