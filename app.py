from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QLineEdit, QMessageBox, QTextEdit, QHBoxLayout, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import sys
import cv2


class VideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Stream Analyzer")
        self.setGeometry(100, 100, 1200, 700)

        # UI Elements
        # Video display area
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")  # Background for QLabel

        # Warnings display area
        self.warning_scroll = QScrollArea()
        self.warning_scroll.setWidgetResizable(True)
        self.warning_container = QWidget()
        self.warning_layout = QVBoxLayout(self.warning_container)
        self.warning_scroll.setWidget(self.warning_container)

        # Input controls
        self.btn_open_file = QPushButton("Open Video File", self)
        self.btn_open_file.clicked.connect(self.open_video_file)

        self.btn_open_rtsp = QPushButton("Open RTSP Stream", self)
        self.btn_open_rtsp.clicked.connect(self.open_rtsp_stream)

        self.rtsp_input = QLineEdit(self)
        self.rtsp_input.setPlaceholderText("Enter RTSP URL here...")

        # Layouts
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.btn_open_file)
        control_layout.addWidget(self.rtsp_input)
        control_layout.addWidget(self.btn_open_rtsp)

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addLayout(control_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.warning_scroll)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Video capture and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Haar cascades for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def open_video_file(self):
        """Open a video file."""
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if video_path:
            self.start_video_stream(video_path)

    def open_rtsp_stream(self):
        """Open an RTSP video stream."""
        rtsp_url = self.rtsp_input.text()
        if rtsp_url:
            self.start_video_stream(rtsp_url)
        else:
            QMessageBox.warning(self, "Input Error", "Please enter a valid RTSP URL!")

    def start_video_stream(self, source):
        """Start capturing the video stream."""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video stream!")
            return
        self.timer.start(30)

    def update_frame(self):
        """Read and analyze the current frame from the video stream."""
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            QMessageBox.information(self, "End of Video", "The video stream has ended.")
            return

        # Analyze the frame (e.g., face detection)
        analyzed_frame, warnings = self.analyze_frame(frame)

        # Display the frame
        self.display_frame(analyzed_frame)

        # Update the warning display
        self.update_warnings(warnings)

    def analyze_frame(self, frame):
        """Analyze the frame and return it with warnings."""
        warnings = []

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped_img = frame[y:y + h, x:x + w]
            warnings.append(("Face detected!", cropped_img))

        return frame, warnings

    def display_frame(self, frame):
        """Display the video frame in the QLabel."""
        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width

        # Convert to QImage
        q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)

        # Scale the QImage to fit the QLabel
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def update_warnings(self, warnings):
        """Update the warning display with detailed text and image details."""
    # Clear previous warnings
        for i in reversed(range(self.warning_layout.count())):
            widget = self.warning_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        for idx, (warning_text, cropped_img) in enumerate(warnings, start=1):
            # Add the warning text with numbering
            detailed_text = f"Object {idx}: {warning_text}"
            warning_label = QLabel(detailed_text)
            warning_label.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
            self.warning_layout.addWidget(warning_label)

            if cropped_img is not None:
                # Convert cropped image to QPixmap
                h, w, ch = cropped_img.shape
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                q_img = QImage(cropped_img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                img_label = QLabel()
                img_label.setPixmap(pixmap)
                self.warning_layout.addWidget(img_label)


    def closeEvent(self, event):
        """Handle the application close event."""
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
