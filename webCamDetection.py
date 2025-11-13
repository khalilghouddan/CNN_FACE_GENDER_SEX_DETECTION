

#biblio OpenCV pour acceder au webcam afficher les image et dessiner les rectangle
import cv2
#pour les opperation mathematique
import numpy as np
#converrt image to Numpy array pour etre trewte par le model
from tensorflow.keras.preprocessing.image import img_to_array
#charger un model pre entrener 
from tensorflow.keras.models import load_model
#simplifier les tache de vision par ordinateur comme la detection d'objet
import cvlib as cv 





def faceDetectionCameraService(model1 , classes1 , model2 , clases2 , webcam , img_width):
    
    #charger le model pre entrener
    model = load_model(model1)

    #open webcam i used 0 for default webcam
    #0 can be replace by 1 or 2 if you have multiple webcam
    webcam = cv2.VideoCapture(webcam)

    if webcam.isOpened() == False:
        print("Error, cannot access webcam")
        exit()
    
    #les class du model
    classes = classes1
        
        
    #webcamfonctionne le programe lit les immage en continu une boucle infinie jusqua ce que je apuit sur q
    while webcam.isOpened():
        
        #status boolean si la lecture a reussi ou non 
        #frame l'image capturer par le webcam matrice numpy
        status , frame =webcam.read()
        
        #fonction de cvlib pour detecter tous les visage dans image
        #utilise un model pre entrener pour la detection base sur (RN SSD +RestNet sous le capot)
        faces , confidence = cv.detect_face(frame)
        
        #boucle pour chaque visage detecter
        #ind indice de visage (0.1.2...)
        #f rectangle de visage (x,y,w,h)
        for idx, f in enumerate(faces):
            
            #recuperation des cordonner du visage
            (x,y,w,h)=f
            
            #desiner un rectangle sur le visage
            #frame = image 
            #(0,255,0) = couleur format RGB
            #2 =epaiseur du rectangle
            cv2.rectangle(frame ,(x,y),(w,h),(0,255,0),2)
            
            #extraire de visage de img
            #np.copy() cree une copie de la region d image
            face_crop = np.copy(frame[y:h,x:w])
            
            #condition pour tail de visage 10pix
            #face_crop.shape = (hauteur , largeur , BGR)
            if(face_crop.shape[0])<10 or (face_crop.shape[1])<10:
                continue
            
            #pretraitement avat prediction 
            #color gray for norgb and 1 dimenson
            face_crop=cv2.cvtColor(face_crop,cv2.COLOR_BGR2GRAY)
    
            #resize the img
            face_crop=cv2.resize(face_crop,(200,200))
            
            # Avant normalisation (96, 96, 3) :
            # [[[ 128,  64, 255], [ 0,  0, 0], ...],
            # [[ 255, 128, 64], ...],
            # ...]
            # Après normalisation (96, 96, 3) :
            # [[[0.502, 0.251, 1.0], [0.0, 0.0, 0.0], ...],
            # [[1.0, 0.502, 0.251], ...],
            # ...]
            face_crop=face_crop.astype("float")/255.0
            
            #convertir image en tableau numpy
            #pour securiser et assure que img est valid pour Tesorflow et keras
            face_crop = img_to_array(face_crop)
            
            #ajouter une dimention suplimentaire pour le batch
            face_crop = np.expand_dims(face_crop ,axis=0)
            
            #prediction du model
            #[0.1 , 0.9] 
            #1ʳᵉ dimension = nombre d’images dans le batch → 1 ici
            # 2ᵉ dimension = nombre de classes
            conf=model.predict(face_crop)[0]
            
            #prendre la class avec la haute confirmation
            #example conf = [0.1 , 0.9]
            #idx -> 1 or 0
            idx = np.argmax(conf)
            
            #prendre classe avec haut confiramtion
            label = classes[idx]
            
            #preparation de text dans affichage
            label ="{}: {:.2f}%".format(label,conf[idx] * 100)
            
            #position de text
            if y - 10 > 10:
                yN = y-10
            else:
                yN=y+10
            
            #ajouter ecriture sur image
            cv2.putText(frame,label,(x,yN) , cv2.FONT_HERSHEY_SIMPLEX ,0.7 ,(0,255,0),2)
            
        #affichage en direct
        cv2.imshow("khalil webcam face detection",frame)
            
        #arreter la baoucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    #liberation de webcan
    webcam.release()
    cv2.destroyAllWindows()
    
#######################################################################################
            
            
            
            
            
faceDetectionCameraService('gender_model.h5', ['Male', 'Female'], '', '', 0, 200 )
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            