# Face Attendance System

"""
Giới thiệu:
    Đây là một ứng dụng chấm công bằng nhận diện khuôn mặt sử dụng PyQt6 để xây dựng giao diện người dùng, TensorFlow để nhận diện khuôn mặt, và OpenCV để xử lý hình ảnh.

Yêu cầu hệ thống:
    - Python 3.x
    - Thư viện cần thiết:
        ```bash
        pip install pyqt6 tensorflow opencv-python pandas numpy
        ```

Cách sử dụng:
    1. Chạy chương trình:
        ```bash
        cd GUI
        python main.py
        ```
    
    2. Các chức năng chính:
        - Nhập Thông Tin: Người dùng nhập thông tin cá nhân (Họ tên, Mã nhân viên) và chụp ảnh khuôn mặt để lưu vào hệ thống.
        - Chấm Công: Nhận diện khuôn mặt của nhân viên và ghi nhận thời gian chấm công.
        - Bảng Chấm Công: Hiển thị danh sách các lần chấm công của nhân viên.

Cấu trúc dự án:
    ```
    .
    ├── dataset/                  # Thư mục chứa ảnh khuôn mặt của nhân viên
    ├── attendance.csv            # File lưu lịch sử chấm công
    ├── face_attendance_model.h5  # Mô hình nhận diện khuôn mặt đã được huấn luyện
    ├── app.py                    # Mã nguồn chính của ứng dụng
    └── README.md                 # Hướng dẫn sử dụng
    ```

Mô tả mã nguồn:
    - `AttendanceDialog`: Hộp thoại nhập thông tin nhân viên.
    - `FaceAttendanceApp`: Giao diện chính của ứng dụng, hiển thị video từ webcam và xử lý nhận diện khuôn mặt.
    - `update_frame()`: Cập nhật hình ảnh từ webcam và thực hiện nhận diện khuôn mặt.
    - `get_user_info()`: Lưu khuôn mặt của nhân viên để huấn luyện mô hình.
    - `record_attendance()`: Lưu thông tin chấm công vào `attendance.csv`.
    - `view_report()`: Hiển thị lịch sử chấm công của nhân viên.

Ghi chú:
    - Cần đảm bảo mô hình `face_attendance_model.h5` đã được huấn luyện trước.
    - Hình ảnh nhân viên phải được lưu vào thư mục `dataset/` để mô hình nhận diện chính xác.

Liên hệ:
    Mọi thắc mắc hoặc góp ý
"""

