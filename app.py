import os
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import model_from_json # pyright: ignore[reportMissingImports]
from pydub import AudioSegment

# --- Constants ---
MODEL_ARCH = "model.json"
MODEL_WEIGHTS = "model_weights.h5"
LABELS_FILE = "emotion_labels.npy"

def load_model():
    with open(MODEL_ARCH, "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(MODEL_WEIGHTS)
    return model

def load_labels():
    # e.g., array([['angry', 'fearful', 'happy', 'neutral', 'sad']], dtype='<U8')
    return np.load(LABELS_FILE, allow_pickle=True)[0].tolist()

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# --- Set up Flask ---
app = Flask(__name__)
CORS(app)
model = load_model()
labels = load_labels()

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio_file found in request."}), 400
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # Save the received file as-is (could be mp3, wav, webm, ogg, etc.)
    temp_uploaded = "temp_uploaded"
    temp_wav = "temp_inference.wav"
    file.save(temp_uploaded)

    try:
        # Convert ANY input to wav using pydub (requires ffmpeg)
        sound = AudioSegment.from_file(temp_uploaded)
        sound = sound.set_channels(1).set_frame_rate(22050)  # librosa's default (mono, 22kHz)
        sound.export(temp_wav, format="wav")
        os.remove(temp_uploaded)

        features = extract_features(temp_wav)
        os.remove(temp_wav)
        if features is None or not np.any(features):
            return jsonify({"error": "Failed to extract features from audio"}), 400

        input_feat = np.expand_dims(features, axis=(0,1))  # Shape: (1, 1, 40)
        probs = model.predict(input_feat)[0]
        idx = int(np.argmax(probs))
        emotion = labels[idx]
        confidence = float(probs[idx])

        return jsonify({
            "emotion": emotion.capitalize(),
            "confidence": confidence
        })

    except Exception as e:
        # Clean up temp files on error
        if os.path.exists(temp_uploaded):
            os.remove(temp_uploaded)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
