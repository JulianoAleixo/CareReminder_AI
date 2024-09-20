# pip install ultralytics opencv-python mediapipe

import cv2
import mediapipe as mp
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

model = YOLO('yolo_model_trained/best.pt')

cap = cv2.VideoCapture(0)


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0):
            inside = not inside
    return inside


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        pill_detected = False
        results_yolo = model(frame)
        for result in results_yolo:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0]
                cls = box.cls[0]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f'{model.names[int(cls)]}: {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if model.names[int(cls)] == 'pill':
                    pill_detected = True
                    pill_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        if results.face_landmarks and pill_detected:
            mouth_landmarks = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                               for idx, landmark in enumerate(results.face_landmarks.landmark)
                               if idx in [17, 146, 375, 37, 267]]

            for landmark in mouth_landmarks:
                cv2.circle(frame, landmark, 2, (0, 0, 255), -1)

            if point_in_polygon(pill_center, mouth_landmarks):
                print("Pílula entrou na boca!")

        cv2.imshow("Camera's Image", frame)

        # Exit when 'ESC' where pressed
        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
