import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import datetime
import os


model = tf.keras.models.load_model("face_attendance_model.h5")


class_names = os.listdir("dataset/")


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


attendance_file = "attendance.csv"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        face = np.expand_dims(face, axis=0) / 255.0

       
        preds = model.predict(face)
        name = class_names[np.argmax(preds)]

       
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([[name, timestamp]], columns=["Name", "Time"])
        df.to_csv(attendance_file, mode="a", header=not os.path.exists(attendance_file), index=False)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
