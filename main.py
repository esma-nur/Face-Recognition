
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
#INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("C:/Users/reyha/python_git/face_recognition/faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv2.CascadeClassifier("C:/Users/reyha/python_git/face_recognition/haarcascade_frontalface_default.xml")
model = pickle.load(open("C:/Users/reyha/python_git/face_recognition/svm_model_160x160.pkl", 'rb'))

cap = cv2.VideoCapture(0)
# WHILE LOOP

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv2.resize(img, (160,160)) # 1x160x160x3
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
        cv2.putText(frame, str(final_name), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow("Face Recognition:", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()