import cv2
import tkinter as tk
from tkinter import filedialog
import threading

def some_condition_met():
    
    
def analyze_video (video_source = "C:/Users/admin/Downloads/yolo/test5.mp4" ):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Phân tích hình ảnh (ví dụ: phát hiện đối tượng)
        # Thay thế đoạn này bằng mã phân tích của bạn
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Hiển thị video đã phân tích
        cv2.imshow("Analyzed Video", gray)

        # Thêm logic cảnh báo nếu cần
        if some_condition_met():  # Thay thế bằng điều kiện thực tế
            send_alert()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def send_alert():
    # Logic gửi cảnh báo (ví dụ: hiển thị trên GUI)
    alert_label.config(text="Alert: Condition met!")

def select_video():
    video_source = filedialog.askopenfilename()
    if video_source:
        threading.Thread(target=analyze_video, args=(video_source,)).start()

# Tạo GUI
root = tk.Tk()
root.title("Video Analysis Application")

select_button = tk.Button(root, text="Select Video", command=select_video)
select_button.pack()

alert_label = tk.Label(root, text="")
alert_label.pack()

root.mainloop()