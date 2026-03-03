import cv2
import mediapipe as mp
import csv

# ---------- COMMON FUNCTION (IMPORTANT) ----------
def extract_landmarks(hand_landmarks):
    data = []
    for lm in hand_landmarks.landmark:
        data.append(lm.x)
        data.append(lm.y)
        data.append(lm.z)
    return data
# -----------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

label = input("Enter sign name (example: hello / thanks / stop / yes / no / emergency): ")

with open("data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # 👇 USE COMMON FUNCTION
            row = extract_landmarks(hand)

            # draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            cv2.putText(
                frame,
                "Press S to Save | Q to Quit",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if cv2.waitKey(1) & 0xFF == ord('s'):
                writer.writerow(row + [label])
                print("Saved:", label)

        cv2.imshow("Collect Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
