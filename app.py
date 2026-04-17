from flask import Flask, request, send_file
import numpy as np
import cv2
import io
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from tf_explain.core.grad_cam import GradCAM

app = Flask(__name__)

model = None

def get_model():
    global model
    if model is None:
        print("Downloading model...", flush=True)
        MODEL_PATH = hf_hub_download(
            repo_id="sorinanekochan/scancerCNN",
            filename="scancer_model_final6.keras"
        )
        print("Loading model...", flush=True)
        model = load_model(MODEL_PATH)
        print("Model ready!", flush=True)
    return model

@app.route("/")
def home():
    return "TF-Explain Grad-CAM API is running"

@app.route("/predict", methods=["POST"])
def predict():
    m = get_model()
    
    file = request.files["image"]
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (128, 128))
    input_img = np.expand_dims(img_resized, axis=0) / 255.0

    preds = m.predict(input_img)

    explainer = GradCAM()
    last_conv_layer_name = None
    for layer in reversed(m.layers):
        if "conv" in layer.name:
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        return "No conv layer found in model", 500

    grid = explainer.explain(
        (input_img, None),
        m,
        class_index=0,
        layer_name=last_conv_layer_name
    )

    heatmap = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode(".png", overlay)
    io_buf = io.BytesIO(buffer)
    return send_file(io_buf, mimetype="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
