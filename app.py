from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import joblib
from gtts import gTTS
from googletrans import Translator

app = Flask(__name__)

VALID_USERNAME = "durga"
VALID_PASSWORD = "12345"

# ================= MODEL LOAD =================
model = joblib.load("sign_model.pkl")
le = joblib.load("label_encoder.pkl")

# ================= MEDIAPIPE SETUP =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

translator = Translator()

def extract_landmarks(hand_landmarks):
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return data


def generate_frames(mode, language="en"):
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        sign_text = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                landmarks = extract_landmarks(hand_landmarks)
                landmarks = np.array(landmarks).reshape(1, -1)

                proba = model.predict_proba(landmarks)[0]
                max_prob = np.max(proba)
                predicted_class = np.argmax(proba)

                if max_prob > 0.75:
                    sign_text = le.inverse_transform([predicted_class])[0]
                else:
                    sign_text = "Not Available"


                if mode == "voice" and sign_text != "Not Available":
                    translated = translator.translate(
                        sign_text, dest=language
                    ).text
                    tts = gTTS(translated, lang=language)
                    tts.save("static/voice.mp3")

                if mode == "text" and sign_text != "Not Available":
                    translated = translator.translate(
                        sign_text, dest=language
                    ).text
                    sign_text = translated

        if sign_text != "":
            cv2.putText(
                frame,
                f"SIGN: {sign_text}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


# ================= ROUTES =================
@app.route("/")
def login():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_user():
    username = request.form.get("username")
    password = request.form.get("password")

    if username == VALID_USERNAME and password == VALID_PASSWORD:
        return {"success": True}
    else:
        return {"success": False}



@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/sign_to_text")
def sign_to_text():
    lang = request.args.get("lang", "en")
    return render_template("sign_to_text.html", lang=lang)


@app.route("/sign_to_voice")
def sign_to_voice():
    lang = request.args.get("lang", "en")
    return render_template("sign_to_voice.html", lang=lang)


@app.route("/video_feed/<mode>/<lang>")
def video_feed(mode, lang):
    return Response(
        generate_frames(mode, lang),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True)