import cv2
import mediapipe as mp
import joblib
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

model = joblib.load("sign_model.pkl")
le = joblib.load("label_encoder.pkl")

cap = cv2.VideoCapture(0)

frame_count = 0
sign = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
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

            prediction = model.predict([landmarks])
            probs = model.predict_proba([landmarks])[0]
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
