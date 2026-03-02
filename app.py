from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from gtts import gTTS
import time
import os

app = Flask(__name__)

# Config (Default users for immediate login)
USERS = {
    "durga": "12345",
    "admin": "admin123"
}
latest_detected_sign = "No Sign"

# Model Load
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model = joblib.load(os.path.join(BASE_DIR, "sign_model.pkl"))
    le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
except Exception as e:
    print("Model load error:", e)
    model = None

@app.route("/")
def index():
    # Show registration page as requested
    return render_template("register.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    
    email = request.form.get("email")
    username = request.form.get("username")
    password = request.form.get("password")
    
    if username in USERS:
        return jsonify({"success": False, "error": "Username already exists."})
    
    # Store user info
    USERS[username] = password
    return jsonify({"success": True})

@app.route("/login_view")
def login_view():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    
    if username in USERS and USERS[username] == password:
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/sign_to_text")
def sign_to_text():
    return render_template("sign_to_text.html", lang="en")

@app.route("/sign_to_voice")
def sign_to_voice():
    return render_template("sign_to_voice.html", lang="en")

@app.route("/learning")
def learning():
    return render_template("learning.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

# ================= AI ASSISTANT LOGIC =================
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
    "yes":       ("Great! I agree with you.", "yes"),
    "no":        ("I understand. No problem.", "no")
}

def get_ai_response(user_input, is_sign=False):
    user_input_lower = user_input.lower().strip()
    if is_sign:
        for key, (reply, gif_key) in SIGN_RESPONSES.items():
            if key in user_input_lower:
                return reply
        return f"I saw your sign: {user_input}. I understand."
    
    if any(greet in user_input_lower for greet in ["hello", "hi"]):
        return "Hello! How can I help you?"
    if "ecosign" in user_input_lower:
        return "EcoSign helps people communicate using sign language."
    return "I'm not sure about that, but I'm learning!"

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    msg = data.get("message", "")
    is_sign = data.get("is_sign", False)
    response = get_ai_response(msg, is_sign)
    return jsonify({"response": response})

@app.route("/api/predict", methods=["POST"])
def predict():
    global latest_detected_sign
    try:
        data = request.json
        landmarks = data.get("landmarks")
        if not landmarks or model is None:
            return jsonify({"sign": "No Sign", "display": "No Sign"})

        input_data = np.array(landmarks).reshape(1, -1)
        proba = model.predict_proba(input_data)[0]
        if np.max(proba) > 0.55:
            predicted_class = np.argmax(proba)
            sign_text = le.inverse_transform([predicted_class])[0]
            if sign_text.lower() in ["hello", "stop", "thanks", "yes", "no", "hi"]:
                latest_detected_sign = sign_text
            return jsonify({"sign": sign_text, "display": sign_text})
        
        return jsonify({"sign": "No Sign", "display": "No Sign"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/speak")
def speak():
    text = request.args.get("text", "")
    if not text: return jsonify({"error": "No text"}), 400
    try:
        tts = gTTS(text, lang='en')
        filename = f"v_{int(time.time())}.mp3"
        filepath = os.path.join("/tmp", filename)
        tts.save(filepath)
        return jsonify({"url": f"/api/voice/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/voice/<filename>")
def serve_voice(filename):
    from flask import send_from_directory
    return send_from_directory("/tmp", filename)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

# Required for Vercel
app = app

