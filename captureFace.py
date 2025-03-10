import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

employee_name = input("Nhập tên nhân viên: ")
save_path = f"dataset/{employee_name}"
os.makedirs(save_path, exist_ok=True)

count = 0
while count < 100:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        cv2.imwrite(f"{save_path}/{count}.jpg", face)
        count += 1

    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(60) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
