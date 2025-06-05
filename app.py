import os
from flask import Flask, request, jsonify
from config import app, MODEL_LOADED, MODEL_PATH, loaded_model
from utils import classify_image, preprocess_image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set maximum file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Brain Disease Diagnosis API is running",
        "model_loaded": MODEL_LOADED,
        "model_format": ".keras"
    })

@app.route('/api/model-info')
def model_info():
    """Get information about the loaded model"""
    if MODEL_LOADED and loaded_model is not None:
        try:
            return jsonify({
                "model_loaded": True,
                "model_format": ".keras",
                "input_shape": str(loaded_model.input_shape),
                "output_shape": str(loaded_model.output_shape),
                "model_path": MODEL_PATH,
                "classes": classification_classes
            })
        except Exception as e:
            return jsonify({"error": f"Error getting model info: {str(e)}"}), 500
    else:
        return jsonify({
            "model_loaded": False,
            "message": "No model loaded"
        }), 503

@app.route("/api/classify", methods=["POST"])
def classify():
    """Classify an uploaded brain scan image"""
    try:
        if not MODEL_LOADED or loaded_model is None:
            return jsonify({"error": "Model not loaded. Please check server configuration."}), 503

        # Validate request
        if "brain_scan" not in request.files:
            return jsonify({"error": "No file uploaded. Please select a brain scan image."}), 400

        brain_scan = request.files["brain_scan"]
        if brain_scan.filename == '':
            return jsonify({"error": "No file selected. Please choose a valid image file."}), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        file_extension = brain_scan.filename.rsplit('.', 1)[1].lower() if '.' in brain_scan.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({"error": "Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)."}), 400

        # Process and classify image
        img_array = preprocess_image(brain_scan)
        classification_result = classify_image(img_array)
        
        return jsonify({
            "success": True,
            "data": classification_result
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Please upload an image smaller than 16MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error. Please try again later."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, host='0.0.0.0')
