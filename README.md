# Face Time Attendance System

"""
Introduction:
This is a face recognition time attendance application using PyQt6 to build the user interface, TensorFlow for face recognition and OpenCV for image processing.

System requirements:
- Python 3.x
- Required libraries:
``` bash
pip install pyqt6 tensorflow opencv-python pandas numpy
```

How to use:
1. Run the program:
``` bash
cd GUI
python main.py
```

2. Main functions:

- Enter information: User enters personal information (Full name, Employee code) and takes a photo of face to save to the system.

- Timekeeping: Recognize employee's face and record time.

- Timekeeping table: Displays a list of employee's timekeeping.

Project configuration:
```
.
├── dataset/ # Folder containing employee face photos
├──attendance.csv # File to save attendance history
├── face_attendance_model.h5 # Trained face recognition model
├── app.py # Main source code of the application
└── README.md # User guide
```

Source code description:
- `AttendanceDialog`: Employee information input dialog.
- `FaceAttendanceApp`: Main interface of the application, displaying video from webcam and face recognition processor.
- `update_frame()`: Update image from webcam and perform face recognition.
- `get_user_info()`: Save employee face code to train the model.
- `record_attendance()`: Save attendance information to `attendance.csv`.

- `view_report()`: Display employee attendance history.

Note:
- Make sure the `face_attendance_model.h5` model has been trained before.

- The image character must be saved to the `dataset/` folder to accurately describe the recognition.

Contact:
Any questions or comments
""