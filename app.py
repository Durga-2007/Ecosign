from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from gtts import gTTS
import time
import os

app = Flask(__name__)

# Config
USERS = {"durga": "12345"}
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
        if np.max(proba) > 0.7:
            predicted_class = np.argmax(proba)
            sign_text = le.inverse_transform([predicted_class])[0]
            if sign_text.lower() in ["hello", "stop", "thanks"]:
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

