import cv2
import mediapipe as mp
import joblib
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

model = joblib.load("sign_model.pkl")
le = joblib.load("label_encoder.pkl")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened() or not cap.read()[0]:
    print("Camera 0 failed. Trying camera 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open any camera. Please check if your webcam is plugged in and NOT being used by another app (like Zoom or Chrome).")
        exit()

frame_count = 0
sign = "Detecting..."

def normalize_landmarks_single(landmarks):
    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
    norm_row = []
    for i in range(0, len(landmarks), 3):
        norm_row.extend([landmarks[i] - wrist_x, landmarks[i+1] - wrist_y, landmarks[i+2] - wrist_z])
    max_val = max(map(abs, norm_row))
    if max_val > 0:
        norm_row = [val / max_val for val in norm_row]
    return norm_row

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame from the camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    frame_count += 1

    # Predict only every 5 frames (faster)
    if frame_count % 5 == 0:
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = []

            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            norm_landmarks = normalize_landmarks_single(landmarks)
            prediction = model.predict([norm_landmarks])
            probs = model.predict_proba([norm_landmarks])[0]
            confidence = max(probs)

            # Only show if confident
            if confidence < 0.6:
                sign = "Not detected"
            else:
                sign = le.inverse_transform(prediction)[0]

    cv2.putText(frame, sign, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
