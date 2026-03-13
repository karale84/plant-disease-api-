import os
import io
import base64
 
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
 
# ── Suppress TensorFlow logs ─────────────────────────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
 
import tensorflow as tf
 
app = Flask(__name__)
CORS(app)
 
# ── 38 PlantVillage class labels ─────────────────────────────────────────────
CLASS_NAMES = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry - Powdery Mildew",
    "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot",
    "Corn - Common Rust",
    "Corn - Northern Leaf Blight",
    "Corn - Healthy",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight",
    "Grape - Healthy",
    "Orange - Huanglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy",
]
 
MODEL = None
MODEL_PATH = os.environ.get("MODEL_PATH", "plant_disease_model.keras")
 
 
# ── Model loader ─────────────────────────────────────────────────────────────
 
def get_model():
    global MODEL
    if MODEL is None:
        print(f"Loading model from: {MODEL_PATH}")
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")
    return MODEL
 
 
# ── Image preprocessing ──────────────────────────────────────────────────────
 
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # shape: (1, 224, 224, 3)
 
 
# ── Routes ───────────────────────────────────────────────────────────────────
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
 
 
@app.route("/predict", methods=["POST"])
def predict():
 
    # ── Parse incoming image ──────────────────────────────────────────────
    try:
        if "file" in request.files:
            image_bytes = request.files["file"].read()
        elif request.is_json and "image" in request.json:
            image_bytes = base64.b64decode(request.json["image"])
        else:
            return jsonify({
                "error": "No image provided. Send a file via multipart/form-data "
                         "or a base64 string in JSON with key 'image'."
            }), 400
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400
 
    # ── Run inference ─────────────────────────────────────────────────────
    try:
        img_array = preprocess(image_bytes)
        model = get_model()
        preds = model.predict(img_array, verbose=0)[0]   # shape: (38,)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500
 
    # ── Format and return results ─────────────────────────────────────────
    top5_idx = np.argsort(preds)[::-1][:5]
    top5 = [
        {
            "label": CLASS_NAMES[i],
            "confidence": round(float(preds[i]) * 100, 2)
        }
        for i in top5_idx
    ]
 
    return jsonify({
        "prediction": top5[0]["label"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
    })
 
 
# ── Entry point ──────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
