import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Import SORT tracker
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from threading import Thread

# Class xử lý video
class VideoProcessor:
    def __init__(self, video_source, text_widget, output_path="test11.mp4"):
        self.video_source = video_source
        self.text_widget = text_widget
        self.running = False
        self.model = YOLO('yolov8n.pt').to("cuda:0")  # Sử dụng mô hình YOLOv8 nhỏ
        self.tracker = Sort()  # Khởi tạo SORT tracker
        self.output_path = output_path  # Đường dẫn video đầu ra (nếu có)
        print(self.model)
    def process_video(self, video_label):
        cap = cv2.VideoCapture(self.video_source)
        
        # Lấy thông tin về kích thước khung hình video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Khởi tạo video writer nếu cần lưu video
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho định dạng MP4
            out = cv2.VideoWriter(self.output_path, fourcc, 30, (width, height))  # 30 fps

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Nhận diện biển số xe trong khung hình
            results = self.model(frame)
            plate_texts = []
            boxes = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Đảm bảo các giá trị tọa độ là số nguyên
                    cropped_plate = frame[y1:y2, x1:x2]
                    gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                    _, thresh_plate = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)
                    plate_text = pytesseract.image_to_string(thresh_plate, config='--psm 8')
                    plate_texts.append(plate_text.strip())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật quanh biển số
                    boxes.append([x1, y1, x2, y2])  # Lưu bounding box để tracker theo dõi

            # Convert boxes to NumPy array
            boxes = np.array(boxes)  # Convert list of boxes to a NumPy array

            # Cập nhật SORT tracker với các bounding box mới
            tracked_objects = self.tracker.update(boxes)  # Truyền bounding boxes cho tracker

            # Hiển thị kết quả nhận diện biển số và ID theo dõi
            if tracked_objects is not None:
                for i in range(len(tracked_objects)):
                    x1, y1, x2, y2, track_id = tracked_objects[i]  # Lấy ID theo dõi
                    cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # Lọc và hiển thị biển số xe đã nhận diện
                    if plate_texts:
                        plate_text = plate_texts[i].replace(' ', '')  # Xóa khoảng trắng
                        cv2.putText(frame, plate_text, (int(x1), int(y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Chuyển đổi khung hình thành RGB và hiển thị lên giao diện
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            video_label.config(image=img)
            video_label.image = img

            # Hiển thị văn bản biển số xe trong text box
            if plate_texts:
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(tk.END, "\n".join(plate_texts))

            # Ghi khung hình vào video đầu ra (nếu có)
            if self.output_path:
                out.write(frame)

            # Thêm độ trễ để video mượt mà hơn
            cv2.waitKey(1)

        cap.release()
        if self.output_path:
            out.release()

    def stop(self):
        self.running = False


# Giao diện người dùng
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện biển số xe")
        self.processor = None

        # Hiển thị ngày giờ và địa điểm
        self.time_label = tk.Label(root, text="", font=("Arial", 12))
        self.time_label.pack(pady=5)
        self.update_time()

        # Nút chọn video và sử dụng camera
        self.btn_select_video = tk.Button(root, text="Chọn video", command=self.select_video)
        self.btn_select_video.pack(pady=5)

        self.btn_use_camera = tk.Button(root, text="Sử dụng camera", command=self.use_camera)
        self.btn_use_camera.pack(pady=5)

        # Khung hiển thị video
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=5)

        # Hộp văn bản hiển thị kết quả nhận diện
        self.text_widget = tk.Text(root, height=5, width=40, font=("Arial", 10))
        self.text_widget.pack(pady=5)

    def update_time(self):
        now = datetime.now()
        day_name = "Chủ Nhật" if now.weekday() == 6 else "Thứ " + str(now.weekday() + 2)
        formatted_time = f"{day_name}, ngày {now.day} tháng {now.month} năm {now.year} {now.strftime('%I:%M %p')}"
        self.time_label.config(text=f"{formatted_time}\nCầu Giấy, Hà Nội, Việt Nam")
        self.root.after(60000, self.update_time)  # Cập nhật mỗi phút

    def select_video(self):
        video_path = filedialog.askopenfilename(title="Chọn video", filetypes=[("Video Files", "*.mp4;*.avi")])
        if video_path:
            self.start_processing(video_path)

    def use_camera(self):
        self.start_processing(0)  # Sử dụng camera mặc định

    def start_processing(self, source):
        if self.processor:
            self.processor.stop()
        self.processor = VideoProcessor(source, self.text_widget, output_path="processed_video.mp4")
        self.processor.running = True
        Thread(target=self.processor.process_video, args=(self.video_label,), daemon=True).start()

    def on_closing(self):
        if self.processor:
            self.processor.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
