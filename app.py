import os
from flask import request, jsonify
from config import app
from utils import classify_image, preprocess_image
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@app.route('/')
def index():
    return "Brain Disease Diagnosis API"

@app.route("/classify", methods=["POST"])
def classify():
    if "xray_image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    xray_image = request.files["xray_image"]
    if xray_image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    img_array = preprocess_image(xray_image)
    classification = classify_image(img_array)
    return jsonify(classification)

if __name__ == "__main__":
    # Use the PORT environment variable set by Render, default to 10000 if not set
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
