
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QMessageBox, QTableWidget, QTableWidgetItem, 
                             QDialog, QFormLayout, QLineEdit)
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, QDateTime
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import time
import sys
import cv2
import os

attendance_file = "attendance.csv"

class AttendanceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nhập Thông Tin Nhân Viên")
        self.setGeometry(300, 300, 300, 200)
        layout = QFormLayout()

        self.name_entry = QLineEdit(self)
        self.id_entry = QLineEdit(self)
        
        layout.addRow("Họ và Tên:", self.name_entry)
        layout.addRow("Mã Nhân Viên:", self.id_entry)
        
        self.submit_button = QPushButton("Xác nhận", self)
        self.submit_button.clicked.connect(self.accept)
        layout.addWidget(self.submit_button)
        self.setLayout(layout)
        
    
    def get_data(self):
        return self.name_entry.text().strip(), self.id_entry.text().strip()

class FaceAttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.name_predict = ""
    
    def initUI(self):
        self.setWindowTitle("Chấm Công Bằng Khuôn Mặt")
        self.setGeometry(200, 200, 700, 500)

        font_button = QFont("Arial", 11, QFont.Weight.Bold)

        self.title_label = QLabel("Chấm Công Bằng Khuôn Mặt", self)
        self.title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(800, 350)
        
        self.info_button = QPushButton("Nhập Thông Tin", self)
        self.info_button.setFont(font_button)
        self.info_button.setStyleSheet("background-color: orange; color: white; padding: 10px; border-radius: 5px;")
        self.info_button.clicked.connect(self.get_user_info)
        
        self.attendance_button = QPushButton("Chấm Công", self)
        self.attendance_button.setFont(font_button)
        self.attendance_button.setStyleSheet("background-color: green; color: white; padding: 10px; border-radius: 5px;")
        self.attendance_button.clicked.connect(self.record_attendance)
        
        self.report_button = QPushButton("Bảng Chấm Công", self)
        self.report_button.setFont(font_button)
        self.report_button.setStyleSheet("background-color: blue; color: white; padding: 10px; border-radius: 5px;")
        self.report_button.clicked.connect(self.view_report)
        
        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.video_label)
        layout.addWidget(self.info_button)
        layout.addWidget(self.attendance_button)
        layout.addWidget(self.report_button)
        
        self.setLayout(layout)
    
    def update_frame(self):

        model = tf.keras.models.load_model("face_attendance_model.h5")

        class_names = os.listdir("dataset/")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        while True:
            ret, frame = self.cap.read()
         
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (128, 128))
                face = np.expand_dims(face, axis=0) / 255.0

                preds = model.predict(face)
                predicted_index = np.argmax(preds)
                confidence = preds[0][predicted_index]
                threshold = 0.5

                if confidence < threshold:
                    name = "Unknown"
                else:
                    name = class_names[predicted_index]

                self.name_predict = name

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)
                h, w, ch = frame.shape
                img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                self.video_label.setPixmap(pixmap)
            if cv2.waitKey(60) & 0xFF == ord("q"):
                break
    
    def get_user_info(self):
        
        dialog = AttendanceDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.name, self.emp_id = dialog.get_data()
        else:
            self.name = ""
            self.emp_id = ""
            
        if self.name or self.emp_id:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            save_path = f"dataset/{self.name}"
            os.makedirs(save_path, exist_ok=True)

            count = 0
            countdown = 3
            last_time = int(time.time())  
            remaining_time = countdown
            while count < 30:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_time = int(time.time())
                if current_time > last_time:
                    remaining_time -= 1
                    last_time = current_time  
               
                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2
               
                overlay = frame.copy()
                cv2.putText(overlay, str(remaining_time), (center_x - 30, center_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10, cv2.LINE_AA)
               
                alpha = remaining_time / countdown
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, (128, 128))
                    cv2.imwrite(f"{save_path}/{count}.jpg", face)
                    count += 1
                cv2.imshow("Lấy Thông Tin Nhân Viên", frame)
                cv2.waitKey(30)
    
        cv2.destroyAllWindows()
        QMessageBox.information(self, "Thành công", f"Lấy thông tin thành công cho {self.name}!")
        self.update_frame()
    
    def record_attendance(self):
        if self.name_predict:
            timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
            data = pd.DataFrame([[self.name_predict, timestamp]], columns=["Name","Time"])
            data.to_csv(attendance_file, mode="a", header=not os.path.exists(attendance_file), index=False, encoding="utf-8")
            QMessageBox.information(self, "Thành công", f"Chấm công thành công cho {self.name_predict}!")
        else:
            QMessageBox.warning(self, "Thất Bại", f"Chưa Nhận Diện Được Người Dùng!")
    
    def view_report(self):
        if not os.path.exists(attendance_file):
            QMessageBox.warning(self, "Lỗi", "Không có dữ liệu chấm công!")
            return

        df = pd.read_csv(attendance_file, on_bad_lines='skip', encoding="utf-8")
        df  = df[df['Name'] == self.name_predict].reset_index(drop=True)
        if df.empty:
            QMessageBox.warning(self, "Lỗi", "Không có dữ liệu hợp lệ trong file!")
            return

        if "Name" not in df.columns or "Time" not in df.columns:
            QMessageBox.warning(self, "Lỗi", "File CSV không đúng định dạng!")
            return

        self.report_window = QWidget()
        self.report_window.setWindowTitle("Thống Kê Chấm Công")
        self.report_window.setGeometry(250, 250, 600, 400)

        table = QTableWidget(self.report_window)
        table.setRowCount(len(df))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Tên", "Thời Gian"])

        for i, row in df.iterrows():
            table.setItem(i, 0, QTableWidgetItem(str(row["Name"])))
            table.setItem(i, 1, QTableWidgetItem(str(row["Time"])))

        layout = QVBoxLayout()
        layout.addWidget(table)
        self.report_window.setLayout(layout)
        self.report_window.show()

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceAttendanceApp()
    window.show()
    sys.exit(app.exec())