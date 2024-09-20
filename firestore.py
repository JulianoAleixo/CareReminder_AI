import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import pytz
import logging
import time

import cv2
import mediapipe as mp
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

model = YOLO('yolo_model_trained/best.pt')

os.environ['GRPC_VERBOSITY'] = 'ERROR'

logging.getLogger('google.auth').setLevel(logging.ERROR)
logging.getLogger('google.auth.transport').setLevel(logging.ERROR)
logging.getLogger('googleapiclient.discovery').setLevel(logging.ERROR)
logging.getLogger('firebase_admin').setLevel(logging.ERROR)

service_account_path = (r"C:\Users\julia\PycharmProjects\pythonProject\10_PillInMouth_YOLOv8_OpenCV_MediaPipe"
                        r"\carereminder-10bab-firebase-adminsdk-eba7k-ad5d86e9b3.json")

if not os.path.exists(service_account_path):
    raise FileNotFoundError(f"Credentials file not found in path: {service_account_path}")

cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred)

db = firestore.client()


def get_current_datetime():
    timezone = pytz.timezone('America/Sao_Paulo')
    now = datetime.now(timezone)
    return now.strftime('%d/%m/%Y'), now.strftime('%H:%M')


def verify_next_compartment():
    docs = db.collection("TabelaRemedios").order_by("compartimento").get()
    sorted_docs = sorted(docs, key=lambda document: int(document.to_dict().get('compartimento', 0)))

    for doc in sorted_docs:
        doc_dict = doc.to_dict()
        compartment = doc_dict.get('compartimento', '')
        scheduled_time = doc_dict.get('horario_previsto', '')
        removed_time = doc_dict.get('horario_tomado', '')

        if scheduled_time != '' and removed_time == '':
            return compartment


def listen_to_compartment(compartment):
    if not compartment:
        print("Invalid Compartment")
        return -1

    while True:
        doc = db.collection('TabelaRemedios').where('compartimento',
                                                    '==', f'{compartment}').get()[0].to_dict()

        removed_time = doc.get('horario_retirado', '')
        if removed_time != '':
            print(f"Medicine removed at time {removed_time}")
            yolo_pill_detection()
            set_taken_time_on_database(compartment)

            print(removed_time)
            break

        time.sleep(15)

    return -1


def set_taken_time_on_database(compartment):
    current_date, current_time = get_current_datetime()

    doc_ref = db.collection('TabelaRemedios').where('compartimento', '==', f'{compartment}').get()[0].reference
    doc_ref.update({
        'horario_tomado': current_date,
        'dia_tomado': current_time
    })
    print(f"Compartment number {compartment} successfully updated!")


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0):
            inside = not inside
    return inside


def yolo_pill_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to Open Camera")
        return False

    medicine_was_taken = False
    print("Camera was open, starting detection...")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and not medicine_was_taken:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
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
                    print("Pill entered the mouth!")
                    cap.release()
                    cv2.destroyAllWindows()
                    medicine_was_taken = True
                    return True

            cv2.imshow("Pill Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return False


compartment_to_verify = -1
while True:
    print(f'compartment: {compartment_to_verify}')
    if compartment_to_verify == -1:
        compartment_to_verify = verify_next_compartment()

        if compartment_to_verify is None:
            print("No compartment to check.")
            break

    print(f"Listening to compartment {compartment_to_verify}...")
    compartment_to_verify = listen_to_compartment(compartment_to_verify)
