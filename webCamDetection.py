import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv

def faceDetectionCameraService(gender_model_path, gender_classes, age_model_path, age_classes, webcam_index=0):
    
    # Load models
    gender_model = load_model(gender_model_path)
    age_model = load_model(age_model_path)
    
    # Open webcam
    webcam = cv2.VideoCapture(webcam_index)
    if not webcam.isOpened():
        print("Error: cannot access webcam")
        return
    
    while webcam.isOpened():
        status, frame = webcam.read()
        if not status:
            break
        
        # Detect faces
        faces, _ = cv.detect_face(frame)
        
        for f in faces:
            (x, y, w, h) = f
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            
            # Crop face
            face_crop = np.copy(frame[y:h, x:w])
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue
            
            # Preprocess for models
            face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop_gray = cv2.resize(face_crop_gray, (200, 200))
            face_crop_gray = face_crop_gray.astype("float") / 255.0
            face_crop_gray = img_to_array(face_crop_gray)
            face_crop_gray = np.expand_dims(face_crop_gray, axis=0)
            
            # Gender prediction
            gender_conf = gender_model.predict(face_crop_gray)[0]
            gender_idx = np.argmax(gender_conf)
            gender_label = "{}: {:.2f}%".format(gender_classes[gender_idx], gender_conf[gender_idx] * 100)
            
            # Age prediction
            age_conf = age_model.predict(face_crop_gray)[0]
            age_idx = np.argmax(age_conf)
            age_label = "{}: {:.2f}%".format(age_classes[age_idx], age_conf[age_idx] * 100)
            
            # Combine labels
            label = f"{gender_label} | {age_label}"
            
            # Position text
            yN = y-10 if y-10 > 10 else y+10
            cv2.putText(frame, label, (x, yN), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Webcam Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()
    cv2.destroyAllWindows()

# Example usage
faceDetectionCameraService(
    'gender_model.h5', ['Male', 'Female'],
    'age_model.h5', ['0-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66+'],
    webcam_index=0
)
