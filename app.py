
from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import joblib
from gtts import gTTS
from googletrans import Translator
import time
import os
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Global for chat detection
latest_detected_sign = "No Sign"


# ================= LOGIN =================
VALID_USERNAME = "durga"
VALID_PASSWORD = "12345"

# ================= MODEL LOAD =================
try:
    model = joblib.load("sign_model.pkl")
    le = joblib.load("label_encoder.pkl")
except Exception as e:
    print("Warning: failed to load ML model or label encoder:", e)
    # Fallback dummy model/label encoder so the Flask app can run without heavy ML deps
    class _DummyModel:
        def predict_proba(self, X):
            # return a low-confidence probability so no sign is detected
            return np.array([[0.0]])

    class _DummyLE:
        def inverse_transform(self, arr):
            return ["No Sign"]

    model = _DummyModel()
    le = _DummyLE()

# ================= MEDIAPIPE =================
try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_draw
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
except Exception as e:
    print("Warning: mediapipe initialization failed:", e)
    mp_hands = None
    hands = None
    mp_draw = None


translator = Translator()

# ================= HELPERS =================
def extract_landmarks(hand_landmarks):
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return data


def draw_text_on_frame(frame, text, position=(20, 40), font_size=40):
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)
        
        # Use Nirmala for Unicode support (Windows)
        font_path = "C:\\Windows\\Fonts\\Nirmala.ttc"
        if not os.path.exists(font_path):
            font_path = "arial.ttf" # Fallback
            
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=(0, 255, 0))
        
        # Convert RGB back to BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("Font rendering error:", e)
        # Fallback to standard OpenCV (English only)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame


def generate_frames(mode, language="en"):
    global latest_detected_sign
    cap = cv2.VideoCapture(0)


    last_sign = ""
    last_spoken_time = 0
    cooldown = 2  # seconds

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if hands is not None:
            result = hands.process(rgb)
        else:
            result = None

        sign_text = "No Sign"

        if result and getattr(result, 'multi_hand_landmarks', None):
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
                    # Specific check for user-requested signs
                    if sign_text.lower() in ["hello", "stop", "thanks", "thank you"]:
                        latest_detected_sign = sign_text


        # ================= SIGN TO TEXT =================
        display_text = sign_text
        if mode == "text" and sign_text != "No Sign":
            try:
                # Use googletrans to translate sign_text for overlay
                # The 'language' variable here is the 'lang' passed from the route
                display_text = translator.translate(sign_text, dest=language).text
            except Exception as e:
                print("Translation error:", e)

        # ================= DISPLAY =================
        if mode == "text":
            frame = draw_text_on_frame(frame, f"SIGN: {display_text}")

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

    cap.release()


# ================= ROUTES =================
@app.route("/")
def login():
    return render_template("login.html")


@app.route("/login", methods=["GET", "POST"])
def login_user():
    if request.method == "GET":
        return render_template("login.html")
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


@app.route("/learning")
def avatar_learning():
    return render_template("learning.html")


@app.route("/video_feed/<mode>/<lang>")
def video_feed(mode, lang):
    return Response(
        generate_frames(mode, language=lang),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )



@app.route("/api/latest_sign")
def get_latest_sign():
    global latest_detected_sign
    sign = latest_detected_sign
    # Clear after read to avoid repeat detections
    latest_detected_sign = "No Sign"
    return {"sign": sign}


@app.route("/api/speak")
def speak():
    text = request.args.get("text", "")
    lang = request.args.get("lang", "en")
    if not text:
        return {"error": "No text"}, 400
    
    try:
        translated = translator.translate(text, dest=lang).text
        tts = gTTS(translated, lang=lang)
        
        # Ensure static dir exists
        if not os.path.exists("static"):
            os.mkdir("static")

        # Unique filename to avoid browser caching
        filename = f"voice_{int(time.time() * 1000)}.mp3"
        filepath = os.path.join("static", filename)
        
        # Clean up old files (optional but good practice)
        for f in os.listdir("static"):
            if f.startswith("voice_") and f.endswith(".mp3"):
                try: os.remove(os.path.join("static", f))
                except: pass
                
        tts.save(filepath)
        return {"url": f"/static/{filename}"}
    except Exception as e:
        print("Speak error:", e)
        return {"error": str(e)}, 500



# ================= SIGN GIF MAP =================
# Maps keywords to locally generated sign images in static/signs/
SIGN_GIFS = {
    "hello":      "/static/signs/hello.png",
    "help":       "/static/signs/help.png",
    "stop":       "/static/signs/stop.png",
    "thank you":  "/static/signs/welcome.png",
    "welcome":    "/static/signs/welcome.png",
    "yes":        "/static/signs/yes.png",
    "no":         "/static/signs/no.png",
    "please":     "/static/signs/please.png",
    "sign":       "/static/signs/sign.png",
}



def get_sign_gif(keyword):
    """Return the GIF URL for a keyword, or None if not available."""
    return SIGN_GIFS.get(keyword.lower())


# ================= SIGN RESPONSE MAP =================
# Maps detected sign labels to (reply_text, reply_gif_key)
SIGN_RESPONSES = {
    "hello":     ("Hello! Good to see you.", "hello"),
    "hi":        ("Hello! Good to see you.", "hello"),
    "stop":      ("Okay. I am stopping here.", "stop"),
    "yes":       ("Yes! I agree.", "yes"),
    "no":        ("No. I understand.", "no"),
    "help":      ("I am here. I will help you.", "help"),
    "please":    ("Of course. I am happy to help.", "please"),
    "thanks":    ("You are welcome.", "welcome"),
    "thank":     ("You are welcome.", "welcome"),
    "welcome":   ("You are welcome.", "welcome"),
    "good":      ("Thank you. That is kind.", None),
    "bad":       ("I am sorry. Let me help you.", None),
    "love":      ("Thank you. That is beautiful.", None),
}


# ================= AI ASSISTANT =================
def get_ai_response(user_input, is_sign=False):
    user_input_lower = user_input.lower().strip()

    # --- If it's a sign detection, check SIGN_RESPONSES first ---
    if is_sign:
        for key, (reply, gif_key) in SIGN_RESPONSES.items():
            if key in user_input_lower:
                return reply, get_sign_gif(gif_key) if gif_key else None
        # Unknown sign detected: show what was detected and acknowledge
        return f"I saw your sign: {user_input}. I understand. Please continue.", get_sign_gif("sign")

    # --- Normal text input ---
    if any(greet in user_input_lower for greet in ["hello", "hi", "hey", "greetings"]):
        return "Hello. How can I help you today?", get_sign_gif("hello")

    if "what is ecosign" in user_input_lower:
        return "ECOSIGN is a platform for easy communication. It helps sign language users.", None

    if "how to use" in user_input_lower:
        return "Go to the dashboard. Choose sign to text or sign to voice. Use your camera to show signs.", get_sign_gif("sign")

    if "stop" in user_input_lower:
        return "Okay I am Stopping here.", get_sign_gif("stop")

    if "thanks" in user_input_lower or "thank you" in user_input_lower:
        return "You're welcome.", get_sign_gif("welcome")

    if "yes" in user_input_lower:
        return "Yes!", get_sign_gif("yes")

    if "no" in user_input_lower:
        return "No.", get_sign_gif("no")

    if "please" in user_input_lower:
        return "Of course. I am happy to help.", get_sign_gif("please")

    if "help" in user_input_lower:
        return "I am here. I will help you.", get_sign_gif("help")

    return "I am not sure. Let me help you find the answer.", None






@app.route("/chat")
def chat_ui():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    user_message = data.get("message", "")
    is_sign = data.get("is_sign", False)
    
    response_text, sign_gif = get_ai_response(user_message, is_sign)
    return {"response": response_text, "sign_gif": sign_gif}




if __name__ == "__main__":
    if not os.path.exists("static"):
        os.mkdir("static")
    app.run(debug=True, threaded=True)

