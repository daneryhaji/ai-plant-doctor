from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os

# Setup
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load model once
model = load_model("plant_disease_model.h5")

# Disease classes
classes = [
    'Corn - Cercospora',
    'Corn - Rust',
    'Corn - Healthy',
    'Peach - Bacterial Spot',
    'Peach - Healthy'
]

# Home
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# About
@app.route('/about')
def about():
    return send_from_directory('.', 'about.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename or "uploaded.png")
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", filename)
    file.save(path)

    try:
        img = imread(path)
        if img.shape[-1] == 4:  # Remove alpha channel
            img = img[:, :, :3]
        img = resize(img, (28, 28))
        img = np.expand_dims(img, axis=0)

        probs = model.predict(img)[0]
        label = classes[np.argmax(probs)]
        confidence = f"{np.max(probs) * 100:.2f}%"

        return jsonify({'prediction': label, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Fallback: serve static files (JS, CSS, images)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

# Run
if __name__ == '__main__':
    app.run()

