from flask_cors import CORS
from flask import Flask, request, jsonify
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    import numpy as np
except ModuleNotFoundError as e:
    raise ImportError("TensorFlow package is not installed. Please install TensorFlow to use this application.") from e
import io
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load your trained model
MODEL_PATH = "fish_disease_classifier.h5"
try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Model input dimensions and class names
IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = [
    "Bacterial diseases-Aeromoniasis",
    "Bacterial gill disease",
    "Bacterial red disease",
    "Fungal diseases-Saprolegniasis",
    "Healthy fish",
    "Parasitic diseases",
    "Viral diseases with tail disease"
]

@app.route("/", methods=["GET"])
def index():
    """
    Default route for the backend API.
    """
    logger.info("Index route accessed.")
    return jsonify({"message": "Welcome to the Fish Disease Prediction API."})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image upload, preprocesses it, and passes it to the model for prediction.
    """
    if "file" not in request.files:
        logger.warning("No file uploaded.")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.warning("No file selected.")
        return jsonify({"error": "No file selected"}), 400

    if not file.content_type.startswith("image/"):
        logger.warning("Invalid file type uploaded.")
        return jsonify({"error": "Invalid file type. Please upload an image."}), 400

    try:
        # Limit file size to 5 MB
        file.stream.seek(0, io.SEEK_END)
        file_size = file.tell()
        if file_size > 5 * 1024 * 1024:  # 5 MB limit
            logger.warning("File size exceeds 5 MB.")
            return jsonify({"error": "File size exceeds the 5 MB limit."}), 400

        file.stream.seek(0)

        # Read and preprocess the image
        image = load_img(io.BytesIO(file.read()), target_size=(IMG_HEIGHT, IMG_WIDTH))
        image_array = img_to_array(image) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)

        # Predict the disease
        if model is None:
            raise ValueError("Model is not loaded.")

        predictions = model.predict(image_array)
        logger.info(f"Raw predictions: {predictions}")

        # Get the class with the highest probability
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        # Return the prediction as JSON
        return jsonify({
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}"
        })

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting the app on port {port}...")
    app.run(debug=True, host="0.0.0.0", port=port)
