from flask_cors import CORS
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io
import os  # Fix: Import os for setting port

app = Flask(__name__)
CORS(app)
# Load your trained model
MODEL_PATH = "fish_disease_classifier.h5"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

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
    return jsonify({"message": "Welcome to the Fish Disease Prediction API."})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image upload, preprocesses it, and passes it to the model for prediction.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.content_type.startswith("image/"):
        return jsonify({"error": "Invalid file type. Please upload an image."}), 400

    try:
        # Read and preprocess the image
        file.stream.seek(0)
        image = load_img(io.BytesIO(file.read()), target_size=(IMG_HEIGHT, IMG_WIDTH))
        image_array = img_to_array(image) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)

        # Predict the disease
        predictions = model.predict(image_array)
        print(f"Raw predictions: {predictions}")

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
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)