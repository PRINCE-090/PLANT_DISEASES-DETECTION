# ================================================================
# app.py — Plant Disease Detection API
# ================================================================
# Model Analysis Results:
#   Architecture : MobileNetV2 (pretrained, fine-tuned)
#   Input        : RGB image, resized to 224×224
#   Output       : 38 plant disease classes (PlantVillage dataset)
#   Classifier   : Linear(1280 → 38)
# ================================================================

import os
import io
import base64
import json
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App Setup ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Class Labels (PlantVillage 38-class dataset) ───────────────
CLASS_NAMES = [
    "Apple — Apple Scab",
    "Apple — Black Rot",
    "Apple — Cedar Apple Rust",
    "Apple — Healthy",
    "Blueberry — Healthy",
    "Cherry — Powdery Mildew",
    "Cherry — Healthy",
    "Corn — Cercospora / Gray Leaf Spot",
    "Corn — Common Rust",
    "Corn — Northern Leaf Blight",
    "Corn — Healthy",
    "Grape — Black Rot",
    "Grape — Esca (Black Measles)",
    "Grape — Leaf Blight",
    "Grape — Healthy",
    "Orange — Huanglongbing (Citrus Greening)",
    "Peach — Bacterial Spot",
    "Peach — Healthy",
    "Pepper Bell — Bacterial Spot",
    "Pepper Bell — Healthy",
    "Potato — Early Blight",
    "Potato — Late Blight",
    "Potato — Healthy",
    "Raspberry — Healthy",
    "Soybean — Healthy",
    "Squash — Powdery Mildew",
    "Strawberry — Leaf Scorch",
    "Strawberry — Healthy",
    "Tomato — Bacterial Spot",
    "Tomato — Early Blight",
    "Tomato — Late Blight",
    "Tomato — Leaf Mold",
    "Tomato — Septoria Leaf Spot",
    "Tomato — Spider Mites",
    "Tomato — Target Spot",
    "Tomato — Yellow Leaf Curl Virus",
    "Tomato — Mosaic Virus",
    "Tomato — Healthy",
]

# Whether the prediction is a disease or a healthy plant
IS_HEALTHY = [
    False, False, False, True,
    True,
    False, True,
    False, False, False, True,
    False, False, False, True,
    False,
    False, True,
    False, True,
    False, False, True,
    True, True, False,
    False, True,
    False, False, False, False, False, False, False, False, False, True,
]

# ── Image Preprocessing ────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Load MobileNetV2 Model ─────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_plant_disease_model.pth')

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 38)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print("✅ Plant disease model loaded successfully!")
    print(f"   Path : {MODEL_PATH}")
    print(f"   Classes: {len(CLASS_NAMES)}")
except FileNotFoundError:
    print(f"⚠️ Model file not found at: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

# ── JSON Logging Function ──────────────────────────────────────
def save_prediction_to_json(data):
    file_path = "prediction_logs.json"

    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logs.append(data)

        with open(file_path, "w") as f:
            json.dump(logs, f, indent=4)

    except Exception as e:
        print("Error saving prediction:", e)

# ── Routes ─────────────────────────────────────────────────────

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "model": "MobileNetV2 — Plant Disease Detector",
        "classes": len(CLASS_NAMES)
    })


@app.route('/predict', methods=['POST'])
def predict():
    print("FILES:", request.files)
    print("FORM:", request.form)
    print("JSON:", request.get_json(silent=True))

    try:

        # ── Read image ──────────────────────────────────────────
        if 'image' in request.files or 'file' in request.files:
            file = request.files.get('image') or request.files.get('file')
            img = Image.open(file.stream).convert('RGB')

        elif request.is_json and 'image' in request.get_json():
            data = request.get_json()
            img_bytes = base64.b64decode(data['image'])
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        else:
            return jsonify({"error": "No image provided."}), 400

        # ── Preprocess ─────────────────────────────────────────
        tensor = preprocess(img).unsqueeze(0)

        # ── Inference ──────────────────────────────────────────
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1).squeeze()

        # ── Top5 predictions ───────────────────────────────────
        top5_probs, top5_indices = torch.topk(probabilities, 5)

        top5 = [
            {
                "label": CLASS_NAMES[idx.item()],
                "confidence": round(prob.item(), 4),
                "is_healthy": IS_HEALTHY[idx.item()]
            }
            for prob, idx in zip(top5_probs, top5_indices)
        ]

        best_idx = top5_indices[0].item()

        result = {
            "prediction": CLASS_NAMES[best_idx],
            "index": best_idx,
            "confidence": round(top5_probs[0].item(), 4),
            "is_healthy": IS_HEALTHY[best_idx],
            "top5": top5
        }

        # ── Save JSON log ──────────────────────────────────────
        save_prediction_to_json(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)