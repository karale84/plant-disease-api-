from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import io
import base64
from PIL import Image

app = Flask(__name__)
CORS(app)

CLASS_NAMES = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot", "Corn - Common Rust", "Corn - Northern Leaf Blight", "Corn - Healthy",
    "Grape - Black Rot", "Grape - Esca (Black Measles)", "Grape - Leaf Blight", "Grape - Healthy",
    "Orange - Huanglongbing (Citrus Greening)",
    "Peach - Bacterial Spot", "Peach - Healthy",
    "Pepper Bell - Bacterial Spot", "Pepper Bell - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Raspberry - Healthy", "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch", "Strawberry - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites", "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic Virus", "Tomato - Healthy"
]

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        import tensorflow as tf
        model_path = os.environ.get("MODEL_PATH", "plant_disease.h5")
        MODEL = tf.keras.models.load_model(model_path, compile=False)
    return MODEL

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            image_bytes = request.files["file"].read()
        elif request.is_json and "image" in request.json:
            image_bytes = base64.b64decode(request.json["image"])
        else:
            return jsonify({"error": "Send a file (multipart) or JSON with image key"}), 400

        img_array = preprocess_image(image_bytes)
        model = get_model()
        preds = model.predict(img_array)[0]

        top5_idx = np.argsort(preds)[::-1][:5]
        results = [
            {"label": CLASS_NAMES[i], "confidence": round(float(preds[i]) * 100, 2)}
            for i in top5_idx
        ]

        return jsonify({
            "prediction": results[0]["label"],
            "confidence": results[0]["confidence"],
            "top5": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
