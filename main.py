from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Class labels
classes = [
    'Corn - Cercospora',
    'Corn - Rust',
    'Corn - Healthy',
    'Peach - Bacterial Spot',
    'Peach - Healthy'
]

# Route: Homepage
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Route: About page
@app.route('/about')
def about():
    return send_from_directory('.', 'about.html')

# Route: Predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename or "image.png")
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    try:
        # Read and preprocess image
        img = imread(filepath)
        if img.shape[-1] == 4:
            img = img[:, :, :3]  # Convert RGBA to RGB
        img = resize(img, (28, 28))
        img = np.expand_dims(img, axis=0)

        # Lazy-load model (prevents memory crashes)
        model = load_model("plant_disease_model.h5")
        prediction_probs = model.predict(img)[0]
        predicted_label = classes[np.argmax(prediction_probs)]
        confidence = f"{np.max(prediction_probs) * 100:.2f}%"

        return jsonify({'prediction': predicted_label, 'confidence': confidence})
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Route: Static files (JS, CSS, etc.)
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

# Run locally (Render uses gunicorn to run `main:app`)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
