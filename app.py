from flask import Flask, render_template, request
import numpy as np
import joblib
from gtts import gTTS
import time
import os

app = Flask(__name__)

# Global for chat detection
latest_detected_sign = "No Sign"

# Vercel doesn't support local camera access from the server.
# We will receive hand landmarks from the client (browser) and predict.


# ================= LOGIN =================
VALID_USERNAME = "durga"
VALID_PASSWORD = "12345"

# ================= MODEL LOAD =================
# Use absolute path for Vercel deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model_path = os.path.join(BASE_DIR, "sign_model.pkl")
    le_path = os.path.join(BASE_DIR, "label_encoder.pkl")
    model = joblib.load(model_path)
    le = joblib.load(le_path)
except Exception as e:
    print("Warning: failed to load ML model or label encoder:", e)
    class _DummyModel:
        def predict_proba(self, X):
            return np.array([[0.0]])
    class _DummyLE:
        def inverse_transform(self, arr):
            return ["No Sign"]
    model = _DummyModel()
    le = _DummyLE()

# Helpers
# Note: extract_landmarks and draw_text_on_frame removed for size optimization

@app.route("/api/health")
def health():
    return {"status": "ok"}


# ================= PREDICTION API (for Vercel) =================
@app.route("/api/predict", methods=["POST"])
def predict_sign():
    global latest_detected_sign
    try:
        data = request.get_json()
        landmarks = data.get("landmarks")
        language = data.get("lang", "en")
        
        if not landmarks:
            return {"sign": "No Sign", "display": "No Sign"}

        # Prepare data for model
        input_data = np.array(landmarks).reshape(1, -1)
        proba = model.predict_proba(input_data)[0]
        max_prob = np.max(proba)
        predicted_class = np.argmax(proba)

        sign_text = "No Sign"
        display_text = "No Sign"

        if max_prob > 0.75:
            sign_text = le.inverse_transform([predicted_class])[0]
            display_text = sign_text
            
            # Update global for chat
            if sign_text.lower() in ["hello", "stop", "thanks", "thank you"]:
                latest_detected_sign = sign_text

            # Translate if needed
            if language != "en":
                try:
                    from googletrans import Translator
                    translator = Translator()
                    display_text = translator.translate(sign_text, dest=language).text
                except Exception as e:
                    print("Translation error:", e)

        return {"sign": sign_text, "display": display_text}
    except Exception as e:
        print("Prediction error:", e)
        return {"error": str(e)}, 500


# The following video_feed route is kept for backward compatibility 
# but won't work on Vercel as intended without local camera access.
# We will use the /predict API for the web-based camera.


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


# Video feed removed for Vercel size limit. Prediction is handled via /api/predict.



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
        from googletrans import Translator
        translator = Translator()
        translated = translator.translate(text, dest=lang).text
        tts = gTTS(translated, lang=lang)
        
        # Vercel filesystem is Read-Only. Use /tmp for transient files.
        filename = f"voice_{int(time.time() * 1000)}.mp3"
        filepath = os.path.join("/tmp", filename)
        
        tts.save(filepath)
        # We will serve this file from a custom route
        return {"url": f"/api/serve-voice/{filename}"}
    except Exception as e:
        print("Speak error:", e)
        return {"error": str(e)}, 500

@app.route("/api/serve-voice/<filename>")
def serve_voice(filename):
    from flask import send_from_directory
    return send_from_directory("/tmp", filename)



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




# No main block needed for Vercel
app = app

