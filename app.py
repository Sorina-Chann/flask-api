from flask import Flask, request, send_file, render_template_string
import numpy as np
import cv2
import io
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download


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
        tf.keras.mixed_precision.set_global_policy('float16')
        model = load_model(MODEL_PATH)
        print("Model ready!", flush=True)
    return model


def grad_cam(m, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=m.input,
        outputs=[m.get_layer(layer_name).output, m.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    return heatmap


HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Grad-CAM Visualizer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: sans-serif;
      background: #f5f5f5;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      min-height: 100vh;
      padding: 2rem 1rem;
    }
    .container {
      background: white;
      border-radius: 12px;
      padding: 2rem;
      width: 100%;
      max-width: 700px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    h1 { font-size: 1.4rem; margin-bottom: 0.4rem; color: #1a1a1a; }
    p.subtitle { font-size: 0.9rem; color: #666; margin-bottom: 1.5rem; }
    .drop-zone {
      border: 2px dashed #ccc;
      border-radius: 10px;
      padding: 2.5rem 1rem;
      text-align: center;
      cursor: pointer;
      transition: background 0.2s, border-color 0.2s;
      background: #fafafa;
    }
    .drop-zone:hover, .drop-zone.drag-over { background: #f0f7ff; border-color: #378ADD; }
    .drop-zone .icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .drop-zone p { font-size: 0.95rem; color: #555; }
    input[type="file"] { display: none; }
    button {
      margin-top: 1rem;
      padding: 0.6rem 1.4rem;
      font-size: 0.95rem;
      border: 1.5px solid #378ADD;
      border-radius: 8px;
      background: white;
      color: #378ADD;
      cursor: pointer;
      transition: background 0.2s, color 0.2s;
    }
    button:hover { background: #378ADD; color: white; }
    button:disabled { opacity: 0.5; cursor: default; }
    #status {
      font-size: 0.85rem; color: #555; margin-top: 0.8rem;
      min-height: 20px; display: flex; align-items: center; gap: 8px;
    }
    .spinner {
      width: 16px; height: 16px;
      border: 2px solid #ddd; border-top-color: #378ADD;
      border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    #error-msg { color: #c0392b; font-size: 0.85rem; margin-top: 0.5rem; }
    .results { display: none; margin-top: 1.5rem; gap: 1rem; grid-template-columns: 1fr 1fr; }
    .results.show { display: grid; }
    .img-card { text-align: center; }
    .img-card .label { font-size: 0.8rem; color: #888; margin-bottom: 6px; }
    .img-card img { width: 100%; border-radius: 8px; border: 1px solid #eee; display: block; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    @media (max-width: 500px) { .results.show { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="container">
  <h1>Grad-CAM Visualizer</h1>
  <p class="subtitle">ارفع صورة وسيعرض النموذج الخريطة الحرارية</p>

  <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
    <div class="icon">📤</div>
    <p>اضغط أو اسحب صورة هنا</p>
    <p style="font-size:0.8rem; color:#aaa; margin-top:4px;">JPG، PNG مدعومة</p>
  </div>
  <input type="file" id="file-input" accept="image/*">

  <div id="preview-area" style="display:none; margin-top:1rem;">
    <div class="img-card">
      <div class="label">الصورة المختارة</div>
      <img id="preview-img" src="" alt="preview" style="max-height:250px; width:auto; margin:auto;">
    </div>
    <div class="actions">
      <button id="analyze-btn" onclick="analyze()">تحليل Grad-CAM ↗</button>
      <button onclick="reset()" style="border-color:#ccc; color:#555;">إعادة تعيين</button>
    </div>
    <div id="status"></div>
    <div id="error-msg"></div>
  </div>

  <div class="results" id="results">
    <div class="img-card">
      <div class="label">الصورة الأصلية</div>
      <img id="orig-img" src="" alt="original">
    </div>
    <div class="img-card">
      <div class="label">الخريطة الحرارية Grad-CAM</div>
      <img id="result-img" src="" alt="heatmap">
    </div>
    <div class="actions" style="grid-column: span 2;">
      <button onclick="downloadResult()">تحميل النتيجة</button>
      <button onclick="reset()" style="border-color:#ccc; color:#555;">تحليل صورة أخرى</button>
    </div>
  </div>
</div>

<script>
  const API_URL = "/predict";
  let selectedFile = null;
  let resultBlob = null;

  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");

  fileInput.addEventListener("change", e => { if (e.target.files[0]) loadFile(e.target.files[0]); });
  dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault(); dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });

  function loadFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      document.getElementById("preview-img").src = ev.target.result;
      document.getElementById("preview-area").style.display = "block";
      document.getElementById("results").classList.remove("show");
      document.getElementById("error-msg").textContent = "";
      document.getElementById("status").innerHTML = "";
    };
    reader.readAsDataURL(file);
  }

  async function analyze() {
    if (!selectedFile) return;
    const btn = document.getElementById("analyze-btn");
    const status = document.getElementById("status");
    const errEl = document.getElementById("error-msg");
    btn.disabled = true;
    errEl.textContent = "";
    status.innerHTML = '<div class="spinner"></div><span>جاري التحليل — قد يستغرق ~60 ثانية في أول طلب...</span>';
    const formData = new FormData();
    formData.append("image", selectedFile);
    try {
      const res = await fetch(API_URL, { method: "POST", body: formData });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "رمز: " + res.status);
      }
      resultBlob = await res.blob();
      const url = URL.createObjectURL(resultBlob);
      document.getElementById("orig-img").src = document.getElementById("preview-img").src;
      document.getElementById("result-img").src = url;
      document.getElementById("preview-area").style.display = "none";
      document.getElementById("results").classList.add("show");
      status.innerHTML = "";
    } catch (err) {
      errEl.textContent = "خطأ: " + err.message;
      status.innerHTML = "";
    }
    btn.disabled = false;
  }

  function reset() {
    selectedFile = null; resultBlob = null; fileInput.value = "";
    document.getElementById("preview-area").style.display = "none";
    document.getElementById("results").classList.remove("show");
    document.getElementById("error-msg").textContent = "";
    document.getElementById("status").innerHTML = "";
  }

  function downloadResult() {
    if (!resultBlob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(resultBlob);
    a.download = "gradcam_result.png";
    a.click();
  }
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        m = get_model()

        file = request.files["image"]
        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (128, 128))
        input_img = np.expand_dims(img_resized, axis=0) / 255.0

        preds = m.predict(input_img)
        print(f"Prediction: {preds}", flush=True)

        last_conv_layer_name = None
        for layer in reversed(m.layers):
            if "conv" in layer.name:
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            return "No conv layer found in model", 500

        print(f"Using layer: {last_conv_layer_name}", flush=True)

        heatmap = grad_cam(m, input_img, last_conv_layer_name)

        heatmap_resized = cv2.resize(heatmap, (128, 128))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)

        _, buffer = cv2.imencode(".png", overlay)
        io_buf = io.BytesIO(buffer)
        return send_file(io_buf, mimetype="image/png")

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return str(e), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
