import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
from sort import Sort
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Load models
vehicle_model = YOLO('yolo11n.pt') 
vehicle_model.to(device) 
plate_model = YOLO('best.pt')
plate_model.to(device)  
ocr = PaddleOCR(use_angle_cls=True, lang='en')  


# Setup video capture and output
cap = cv2.VideoCapture('test5.mp4')
out = cv2.VideoWriter(
    'test5_output.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    int(cap.get(cv2.CAP_PROP_FPS)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

# Initialize SORT tracker
tracker = Sort()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    for v_box in vehicle_model.predict(frame,device=device)[0].boxes.data.cpu().numpy():
        if v_box[5] in [2, 3]:  # Only cars or motorcycles
            vehicle_img = frame[int(v_box[1]):int(v_box[3]), int(v_box[0]):int(v_box[2])]
            frame = cv2.rectangle(frame, (int(v_box[0]), int(v_box[1])), (int(v_box[2]), int(v_box[3])), (0, 255, 0), 2)

            # Detect the license plate and perform OCR
            for p_box in plate_model.predict(vehicle_img,device=device)[0].boxes.data.cpu().numpy():
                plate_img = vehicle_img[int(p_box[1]):int(p_box[3]), int(p_box[0]):int(p_box[2])]

                # Sử dụng PaddleOCR để nhận diện ký tự
                ocr_results = ocr.ocr(plate_img, cls=True)

                # Kiểm tra nếu không có kết quả OCR hoặc ocr_results là None
                if ocr_results and ocr_results[0]:
                    text = " ".join([result[1][0] for result in ocr_results[0]])  # Ghép các ký tự lại
                    frame = cv2.putText(frame, text, (int(v_box[0]), int(v_box[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    text = "Không nhận diện được biển số"

                # Vẽ kết quả nhận diện lên khung hình
                frame = cv2.rectangle(frame, 
                                      (int(p_box[0] + v_box[0]), int(p_box[1] + v_box[1])),
                                      (int(p_box[2] + v_box[0]), int(p_box[3] + v_box[1])), 
                                      (0, 0, 255), 2)

            detections.append([v_box[0], v_box[1], v_box[2], v_box[3], 1])  # [x1, y1, x2, y2, score]

    if len(detections) > 0:
        detections = np.array(detections)
        trackers = tracker.update(detections)  # Update the tracker with the new detections

        for track in trackers:
            x1, y1, x2, y2, track_id = track
            # Display the tracking ID below the bounding box
            frame = cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y2) + 20),  # Position the ID below the box
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # Draw the bounding box for the tracked object
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    out.write(frame)  # Write the frame with drawn bounding boxes to the output video
    new_width, new_height = 1080, 720
    resized_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow('Video',resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
print("Video đã xử lý được lưu thành công!")
