import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import keras.backend as K
import cv2
import numpy as np
from flask_cors import CORS
import logging
logging.basicConfig(level=logging.INFO)

# Config
app = Flask(__name__)
CORS(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# Load Model
model = None


def load_model():
    global model
    model_path = os.path.join(os.getcwd(), 'model.h5')
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_score': f1_score, 'f2_score': f2_score})
    logging.info('Model loaded')

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def f2_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f2_val = 5 * (precision * recall) / (4 * precision + recall + K.epsilon())
    return f2_val

# Image Processing
def preprocess_image(img_path, img_dim=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot find or open the image at path: {img_path}")
    img = cv2.resize(img, img_dim)
    img = img / 255.0
    return img



def predict_image(img_path):
    try:
        img = preprocess_image(img_path)
        prediction = model.predict(np.expand_dims(img, axis=0))
        return prediction[0][0]
    except Exception as e:
        logging.error(f"Failed to predict image at {img_path}: {str(e)}")
        raise


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# API Endpoints
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({"error": f"Cannot find {filename}"}), 404


import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.route('/predict', methods=['POST'])
def predict():
    logging.info('Prediction request received')

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logging.info(f"Attempting to save file to: {file_path}")
        file.save(file_path)
        logging.info(f"File saved to: {file_path}")

        try:
            pred_value = predict_image(file_path)
            label = "ANOMALY" if pred_value > 0.5 else "NORMAL"
            probability = pred_value if label == "ANOMALY" else (1 - pred_value)
            image_base64 = image_to_base64(file_path)
            return jsonify(
                {"filename": filename, "prediction": label, "probability": float(probability), "image": image_base64})

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": f"Error during prediction: {str(e)}"}), 500


        finally:
            if os.path.exists(file_path):
                logging.info(f"Deleting file at: {file_path}")
                #os.remove(file_path)
            else:
                logging.warning(f"File not found at: {file_path}")

    return jsonify({"error": "Invalid file type"}), 400


# Main Execution
if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
