import os
import gdown
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
import base64
import io

app = Flask(__name__)


MODEL_PATH = "scancer_model_final6.keras"

FILE_ID = "1eA69ZinKpU0Ycn4GvmEIcU4dsG4Tt9zw"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


IMG_SIZE = (128, 128)
LAYER_NAME = "conv2d_1"

def preprocess(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)


HTML = """
<!doctype html>
<title>Grad-CAM Test</title>

<h2>Upload Image for Grad-CAM</h2>

<form method=post enctype=multipart/form-data>
  <input type=file name=image>
  <input type=submit value="Run">
</form>

{% if prediction is not none %}
    <h3>Prediction: {{ prediction }}</h3>
    <h3>Confidence: {{ confidence }}</h3>
    <img src="data:image/png;base64,{{ heatmap }}" width="400">
{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    heatmap_base64 = None

    if request.method == "POST":

        file = request.files["image"]
        img = Image.open(file).convert("RGB")

        x = preprocess(img)

     
        preds = model.predict(x)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds))

        prediction = class_idx

     
        explainer = GradCAM()

        grid = explainer.explain(
            validation_data=(x, None),
            model=model,
            class_index=class_idx,
            layer_name=LAYER_NAME
        )

     
        heatmap_img = Image.fromarray(grid)

        buffer = io.BytesIO()
        heatmap_img.save(buffer, format="PNG")
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()

    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        heatmap=heatmap_base64
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
