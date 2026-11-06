from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Tomato_AI_Model.keras")
IMAGE_SIZE = (224, 224)

# This order MUST match the folder/class order used while training the model.
CLASS_NAMES = [
    "Tomato Bacterial spot",
    "Tomato early blight",
    "Tomato late blight",
    "Tomato healthy",
]

# Map the model's class labels to diagnosis IDs already understood by crop-monitor.html.
ADVISORY_IDS = {
    "Tomato Bacterial spot": "bacterial",
    "Tomato early blight": "leafspot",
    "Tomato late blight": "lateblight",
    "Tomato healthy": "healthy",
}

app = Flask(__name__)
CORS(app)  # Allows the HTML page to call this API from another local origin.

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Put Tomato_AI_Model.keras beside app.py."
    )

model = load_model(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({
        "status": "Tomato AI API is running",
        "endpoint": "/predict",
        "classes": CLASS_NAMES,
    })

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image received. Send the photo in the 'image' field."}), 400

    uploaded = request.files["image"]
    try:
        image = Image.open(uploaded.stream).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32)
        # The trained model contains its own Rescaling layer, so keep pixels in 0..255.
        batch = np.expand_dims(array, axis=0)
        probabilities = model.predict(batch, verbose=0)[0]

        if len(probabilities) != len(CLASS_NAMES):
            return jsonify({"error": "Model output count does not match the four configured classes."}), 500

        index = int(np.argmax(probabilities))
        class_name = CLASS_NAMES[index]
        confidence = float(probabilities[index])

        return jsonify({
            "label": ADVISORY_IDS[class_name],
            "class_name": class_name,
            "confidence": confidence,
            "all_scores": {
                CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(CLASS_NAMES))
            },
            "message": "AI result is a suggestion. Confirm the symptoms before taking action.",
        })
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Could not read the image. Please upload a JPG or PNG photo."}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500

if __name__ == "__main__":
    # Local testing only. For mobile access on the same Wi-Fi, use the laptop's LAN IP.
    app.run(host="0.0.0.0", port=5000, debug=False)
