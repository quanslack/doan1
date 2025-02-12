import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QLineEdit, QMessageBox, QHBoxLayout, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import sys
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
from sort import Sort
import torch


class DualVideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dual Video Stream Analyzer")
        self.setGeometry(100, 100, 1600, 900)

        # UI Elements
        # Video display areas
        self.video_label1 = QLabel(self)
        self.video_label1.setAlignment(Qt.AlignCenter)
        self.video_label1.setStyleSheet("background-color: black;")
        self.video_label1.setFixedSize(800, 600)

        self.video_label2 = QLabel(self)
        self.video_label2.setAlignment(Qt.AlignCenter)
        self.video_label2.setStyleSheet("background-color: black;")
        self.video_label2.setFixedSize(800, 600)

        # Warnings display area
        self.warning_scroll = QScrollArea()
        self.warning_scroll.setWidgetResizable(True)
        self.warning_scroll.setFixedWidth(300)  # Smaller width for the warnings panel
        self.warning_container = QWidget()
        self.warning_layout = QVBoxLayout(self.warning_container)
        self.warning_scroll.setWidget(self.warning_container)
        self.warning_scroll.setMinimumWidth(1600)
        self.warning_scroll.setMaximumWidth(1300)
        self.warning_layout.setSpacing(10)


        # Input controls
        self.btn_open_file1 = QPushButton("Open Video File 1", self)
        self.btn_open_file1.clicked.connect(lambda: self.open_video_file(1))

        self.btn_open_file2 = QPushButton("Open Video File 2", self)
        self.btn_open_file2.clicked.connect(lambda: self.open_video_file(2))

        self.rtsp_input1 = QLineEdit(self)
        self.rtsp_input1.setPlaceholderText("Enter RTSP URL 1 here...")

        self.rtsp_input2 = QLineEdit(self)
        self.rtsp_input2.setPlaceholderText("Enter RTSP URL 2 here...")

        self.btn_open_rtsp1 = QPushButton("Open RTSP Stream 1", self)
        self.btn_open_rtsp1.clicked.connect(lambda: self.open_rtsp_stream(1))

        self.btn_open_rtsp2 = QPushButton("Open RTSP Stream 2", self)
        self.btn_open_rtsp2.clicked.connect(lambda: self.open_rtsp_stream(2))

        # Layouts
        control_layout1 = QHBoxLayout()
        control_layout1.addWidget(self.btn_open_file1)
        control_layout1.addWidget(self.rtsp_input1)
        control_layout1.addWidget(self.btn_open_rtsp1)

        control_layout2 = QHBoxLayout()
        control_layout2.addWidget(self.btn_open_file2)
        control_layout2.addWidget(self.rtsp_input2)
        control_layout2.addWidget(self.btn_open_rtsp2)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_label1)
        video_layout.addWidget(self.video_label2)

        main_layout = QVBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addLayout(control_layout1)
        main_layout.addLayout(control_layout2)
        main_layout.addWidget(self.warning_scroll)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Video capture and timers
        self.caps = [None, None]
        self.timers = [QTimer(self), QTimer(self)]
        self.timers[0].timeout.connect(lambda: self.update_frame(1))
        self.timers[1].timeout.connect(lambda: self.update_frame(2))

        # Haar cascades for face detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.vehicle_model = YOLO('yolo11n.pt') 
        self.vehicle_model.to(self.device) 
        self.plate_model = YOLO('best.pt')
        self.plate_model.to(self.device)  
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.tracker = Sort()
        self.tracked_id = []

    def open_video_file(self, idx):
        """Open a video file."""
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if video_path:
            self.start_video_stream(video_path, idx)

    def open_rtsp_stream(self):
            """Open an RTSP video stream."""
            rtsp_url = self.rtsp_input.text()
            if rtsp_url:
                self.start_video_stream(rtsp_url)
            else:
                QMessageBox.warning(self, "Input Error", "Please enter a valid RTSP URL!")

    def start_video_stream(self, source, idx):
        """Start capturing the video stream."""
        if self.caps[idx - 1]:
            self.caps[idx - 1].release()

        self.caps[idx - 1] = cv2.VideoCapture(source)
        if not self.caps[idx - 1].isOpened():
            QMessageBox.critical(self, "Error", f"Failed to open video stream {idx}!")
            return

        self.timers[idx - 1].start(30)


    def update_frame(self, idx):
        """Read and analyze the current frame from the video stream."""
        if self.caps[idx - 1] is None or not self.caps[idx - 1].isOpened():
            return

        ret, frame = self.caps[idx - 1].read()
        if not ret:
            self.timers[idx - 1].stop()
            self.caps[idx - 1].release()
            QMessageBox.information(self, "End of Video", f"The video stream {idx} has ended.")
            return

        # Analyze the frame
        analyzed_frame, warnings = self.analyze_frame(frame)

        # Display the frame in the appropriate QLabel
        video_label = self.video_label1 if idx == 1 else self.video_label2
        self.display_frame(analyzed_frame, video_label)

        # Update the warning display for this stream
        self.update_warnings(warnings, idx)


        
    def analyze_frame(self, frame):
        """Analyze the frame and return it with warnings."""
        warnings = []

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []
        for v_box in self.vehicle_model.predict(frame,device=self.device,conf=0.4)[0].boxes.data.cpu().numpy():
            if v_box[5] in [2, 3]:  # Only cars or motorcycles
                detections.append([v_box[0], v_box[1], v_box[2], v_box[3], 1])  # [x1, y1, x2, y2, score]

        if len(detections) > 0:
            detections = np.array(detections)
            trackers = self.tracker.update(detections)  # Update the tracker with the new detections

            for track in trackers:
                x1, y1, x2, y2, track_id = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Display the tracking ID below the bounding box
                frame = cv2.putText(frame, f'ID: {int(track_id)}', (x1, y2 + 20),  # Position the ID below the box
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                text = ""
                # if int(track_id) not in self.tracked_id and (x1<x2) and (y1<y2) and (x1>0) and (y1>0):
                if (x1<x2) and (y1<y2) and (x1>0) and (y1>0):
                    vehicle_img = frame[y1:y2, x1:x2]
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Detect the license plate and perform OCR
                    print(x1, y1, x2, y2)
                    for p_box in self.plate_model.predict(vehicle_img,device=self.device)[0].boxes.data.cpu().numpy():
                        plate_img = vehicle_img[int(p_box[1]):int(p_box[3]), int(p_box[0]):int(p_box[2])]

                        # Sử dụng PaddleOCR để nhận diện ký tự
                        ocr_results = self.ocr.ocr(plate_img, cls=True)

                        # Kiểm tra nếu không có kết quả OCR hoặc ocr_results là None
                        if ocr_results and ocr_results[0]:
                            text = " ".join([result[1][0] for result in ocr_results[0]])  # Ghép các ký tự lại
                            frame = cv2.putText(frame, text, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        else:
                            text = "Không nhận diện được biển số"

                        # Vẽ kết quả nhận diện lên khung hình
                        frame = cv2.rectangle(frame, 
                                            (int(p_box[0] + x1), int(p_box[1] + y1)),
                                            (int(p_box[2] + x1), int(p_box[3] + y1)), 
                                            (0, 0, 255), 2)
                    if int(track_id) not in self.tracked_id:
                        self.tracked_id.append(int(track_id))
                        warnings.append(("Found a new vehicle, Id: " + str(track_id) + " " + text, vehicle_img))

        return frame, warnings

    def display_frame(self, frame, video_label):
        """Display the video frame in the QLabel."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width
        q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            video_label.width(), video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        video_label.setPixmap(scaled_pixmap)


    def update_warnings(self, warnings, idx):
        """Update the warning display for a specific stream."""
        layout = self.warning_layout  # You can add logic to handle separate layouts for different streams

        # Clear the existing warnings for this stream
        # while layout.count() > 0:
        #     item = layout.takeAt(0)
        #     widget = item.widget()
        #     if widget is not None:
        #         widget.deleteLater()

        # Add the new warnings
        for warning_text, cropped_img in warnings:
            warning_label = QLabel(warning_text)
            warning_label.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
            layout.addWidget(warning_label)

            if cropped_img is not None:
                h, w, ch = cropped_img.shape
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                q_img = QImage(cropped_img_rgb.data, w, h, ch * w, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_img).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                img_label = QLabel()
                img_label.setPixmap(pixmap)
                layout.addWidget(img_label)


    def closeEvent(self, event):
        """Handle the application close event."""
        for cap in self.caps:
            if cap:
                cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DualVideoAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
