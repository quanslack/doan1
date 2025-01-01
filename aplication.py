import sys
import numpy as np
import cv2
import pytesseract
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class VideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Analyzer with Bounding Box Details")
        self.setGeometry(100, 100, 1200, 700)

        # UI Elements
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        self.warning_panel = QLabel(self)
        self.warning_panel.setAlignment(Qt.AlignLeft)
        self.warning_panel.setStyleSheet("background-color: lightyellow; border: 1px solid black; font-size: 12px;")
        self.warning_panel.setText("No warnings yet.")

        self.btn_open_file = QPushButton("Open Video File", self)
        self.btn_open_file.clicked.connect(self.open_video_file)

        self.btn_open_rtsp = QPushButton("Open RTSP Stream", self)
        self.btn_open_rtsp.clicked.connect(self.open_rtsp_stream)

        self.rtsp_input = QLineEdit(self)
        self.rtsp_input.setPlaceholderText("Enter RTSP URL here...")

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.btn_open_file)
        control_layout.addWidget(self.rtsp_input)
        control_layout.addWidget(self.btn_open_rtsp)

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addLayout(control_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.warning_panel, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Video capture and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Initialize Tesseract OCR
        pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Update path if necessary

    def open_video_file(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if video_path:
            self.start_video_stream(video_path)

    def open_rtsp_stream(self):
        rtsp_url = self.rtsp_input.text()
        if rtsp_url:
            self.start_video_stream(rtsp_url)

    def start_video_stream(self, source):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.warning_panel.setText("Error: Failed to open video stream!")
            return
        self.timer.start(30)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            self.warning_panel.setText("End of video stream.")
            return

        analyzed_frame, warnings = self.analyze_frame(frame)
        self.display_frame(analyzed_frame)
        self.update_warning_panel(warnings)

    def analyze_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_detected = self.detect_motion(gray_frame)
        warnings = []

        if motion_detected:
            text_regions = self.detect_text(frame)
            for (x, y, w, h, text) in text_regions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                warnings.append(text)

        return frame, warnings

    def detect_motion(self, gray_frame):
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = gray_frame
            return False

        frame_diff = cv2.absdiff(self.prev_frame, gray_frame)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        self.prev_frame = gray_frame
        return np.sum(thresh) > 5000

    def detect_text(self, frame):
        text_regions = []
        h, w, _ = frame.shape
        ocr_data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)

        for i, text in enumerate(ocr_data['text']):
            if text.strip():
                x, y, width, height = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                text_regions.append((x, y, width, height, text))

        return text_regions

    def display_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        q_img = QImage(rgb_frame.data, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    def update_warning_panel(self, warnings):
        if warnings:
            self.warning_panel.setText("\n".join(warnings))
        else:
            self.warning_panel.setText("No warnings.")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
