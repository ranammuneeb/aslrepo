from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
from gtts import gTTS
import uuid
from datetime import datetime, timedelta
import glob
import gdown
import difflib


WORDS_FILE = "words.txt"
WORD_LIST = []
if os.path.exists(WORDS_FILE):
    with open(WORDS_FILE, "r") as f:
        WORD_LIST = [w.strip().lower() for w in f.readlines()]
else:
    print("  Warning: words.txt not found. Suggestions will be empty.")

# -------------------------------------------------
app = Flask(__name__)

predicted_sentence = ""
current_sign = ""
prediction_count = 0
threshold_frames = 15
last_prediction = ""

AUDIO_DIR = os.path.join("static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


MODEL_PATH = "model/asl_model.joblib"
ENCODER_PATH = "model/label_encoder.joblib"

os.makedirs("model", exist_ok=True)





model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

print("Model and Encoder loaded")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global predicted_sentence, current_sign
        global prediction_count, last_prediction

        ret, frame = self.video.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                x_input = np.array(features).reshape(1, -1)
                y_pred = model.predict(x_input)
                label = le.inverse_transform(y_pred)[0]

                current_sign = label

                if label == last_prediction:
                    prediction_count += 1
                else:
                    prediction_count = 0
                    last_prediction = label

                if prediction_count == threshold_frames:
                    if label == "space":
                        predicted_sentence += " "
                    elif label == "del":
                        predicted_sentence = predicted_sentence[:-1]
                    elif label != "nothing":
                        predicted_sentence += label
                    prediction_count = 0
        else:
            current_sign = "nothing"

        # Overlays are now handled by HTML/CSS in the frontend for a cleaner look
        # cv2.rectangle(frame, (10, 10), (630, 100), (0, 0, 0), -1)
        # cv2.putText(frame, f"Sign: {current_sign}", (20, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # cv2.putText(frame, f"Sentence: {predicted_sentence}", (20, 80),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes()

    def get_suggestions(self):
        global predicted_sentence
        if not predicted_sentence:
            return []
        
        words = predicted_sentence.strip().split(" ")
        if not words:
            return []
            
        last_word = words[-1].lower()
        if not last_word:
            return []
            
        matches = difflib.get_close_matches(last_word, WORD_LIST, n=4, cutoff=0.3)
        
        starts_with = [w for w in WORD_LIST if w.startswith(last_word)][:2]
        
        all_suggestions = list(set(matches + starts_with))
        return all_suggestions[:4]


def gen_frames():
    global camera_active
    camera = None
    try:
        while True:
            if camera_active:
                if camera is None:
                    camera = VideoCamera()
                frame = camera.get_frame()
                if frame:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" +
                           frame + b"\r\n")
                else:
                    break
            else:
                if camera is not None:
                    del camera
                    camera = None
                break
            time.sleep(1 / 30)
    except Exception as e:
        print(f"Error in gen_frames: {e}")
    finally:
        if camera is not None:
            del camera
            camera = None


# -------------------------------------------------
# Audio Helpers
# -------------------------------------------------
def cleanup_old_audio_files():
    cutoff = datetime.now() - timedelta(hours=1)
    for f in glob.glob(os.path.join(AUDIO_DIR, "*.mp3")):
        if datetime.fromtimestamp(os.path.getctime(f)) < cutoff:
            os.remove(f)

def generate_audio_file(text):
    if not text.strip():
        return None

    cleanup_old_audio_files()

    filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    tts = gTTS(text=text, lang="en")
    tts.save(filepath)

    return filename

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/get_sentence")
def get_sentence():
    suggestions = []
    if predicted_sentence:
        words = predicted_sentence.strip().split(" ")
        if words:
            last_word = words[-1].lower()
            if last_word:
                 
                matches = difflib.get_close_matches(last_word, WORD_LIST, n=4, cutoff=0.4)
                # Starts with
                starts_with = [w for w in WORD_LIST if w.startswith(last_word)][:4]
                
                
                suggestions = list(set(starts_with + matches))
                
                suggestions = [s for s in suggestions if s != last_word]
                suggestions = suggestions[:4]

    return jsonify({
        "sentence": predicted_sentence,
        "current_sign": current_sign,
        "suggestions": suggestions
    })

@app.route("/clear_sentence", methods=["POST"])
def clear_sentence():
    global predicted_sentence
    predicted_sentence = ""
    return jsonify({"status": "cleared"})

@app.route("/add_word", methods=["POST"])
def add_word():
    global predicted_sentence
    data = request.json
    word_to_add = data.get("word")
    
    if not word_to_add:
        return jsonify({"error": "No word provided"}), 400
        
    
    words = predicted_sentence.strip().split(" ")
    if words:
        words[-1] = word_to_add
        predicted_sentence = " ".join(words) + " " 
    else:
        predicted_sentence = word_to_add + " "
        
    return jsonify({"sentence": predicted_sentence})

@app.route("/speak_sentence", methods=["POST"])
def speak_sentence():
    filename = generate_audio_file(predicted_sentence)
    if not filename:
        return jsonify({"error": "No text"}), 400

    return jsonify({
        "audio_url": f"/static/audio/{filename}",
        "sentence": predicted_sentence
    })

@app.route("/static/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

camera_active = False

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    return jsonify({"status": "started"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    print("Starting Flask ASL App")
    app.run(host="0.0.0.0", port=5000, debug=True)
