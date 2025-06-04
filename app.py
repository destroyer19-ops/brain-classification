import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__, static_folder='static')
CORS(app)

# Load your .keras model
# MODEL_PATH = "./model/MobileNet_Parkinson_diagnosis.keras"  # Update this path to your actual model
MODEL_PATH = os.environ.get("MODEL_PATH", "./model/MobileNet_Parkinson_diagnosis.keras")
try:
    # Load the .keras model
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    print(f"Model input shape: {loaded_model.input_shape}")
    print(f"Model output shape: {loaded_model.output_shape}")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print("Using mock model for demonstration...")
    
    # Fallback mock model for testing
    class MockModel:
        def predict(self, image, verbose=0):
            return np.array([[0.1, 0.7, 0.2]])  # [Alzheimer's, Normal, Parkinson's]
        
        @property
        def input_shape(self):
            return (None, 224, 224, 3)
        
        @property
        def output_shape(self):
            return (None, 3)
    
    loaded_model = MockModel()
    MODEL_LOADED = False

classification_classes = {
    0: 'Alzheimer\'s Disease',
    1: 'Normal',
    2: 'Parkinson\'s Disease',
}

def preprocess_image(image_file):
    """
    Preprocess the input image for classification
    
    Parameters:
    image_file: The uploaded image file
    
    Returns:
    np.array: Preprocessed image array
    """
    try:
        # Open and convert image
        image = Image.open(image_file).convert("RGB")
        
        # Get model input shape (assuming it's (batch, height, width, channels))
        if hasattr(loaded_model, 'input_shape'):
            input_shape = loaded_model.input_shape
            if len(input_shape) == 4:  # (batch, height, width, channels)
                target_size = (input_shape[1], input_shape[2])
            else:
                target_size = (224, 224)  # Default fallback
        else:
            target_size = (224, 224)  # Default fallback
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to array
        image = img_to_array(image)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Normalize pixel values (adjust based on your model's training)
        # Common normalizations:
        image = image / 255.0  # Scale to [0, 1]
        # OR: image = (image - 127.5) / 127.5  # Scale to [-1, 1]
        # OR: Use ImageNet normalization if your model was trained with it
        
        return image
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def classify_image(image_array):
    """
    Classify the preprocessed image
    
    Parameters:
    image_array: Preprocessed image array
    
    Returns:
    dict: Classification results with probabilities
    """
    try:
        # Make prediction
        prediction = loaded_model.predict(image_array, verbose=0)[0]
        
        # Get the predicted class
        predicted_class_idx = np.argmax(prediction)
        classified_label = classification_classes[predicted_class_idx]
        confidence = float(np.max(prediction))
        
        # Ensure probabilities sum to 1 (apply softmax if needed)
        if not np.isclose(np.sum(prediction), 1.0, atol=1e-3):
            prediction = tf.nn.softmax(prediction).numpy()
        
        return {
            "classification": classified_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "alzheimers": round(float(prediction[0]), 4),
                "normal": round(float(prediction[1]), 4),
                "parkinsons": round(float(prediction[2]), 4)
            },
            "model_info": {
                "format": ".keras",
                "loaded": MODEL_LOADED,
                "input_shape": str(loaded_model.input_shape) if hasattr(loaded_model, 'input_shape') else "Unknown"
            }
        }
        
    except Exception as e:
        raise ValueError(f"Error during classification: {str(e)}")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

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
    if MODEL_LOADED:
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
            "message": "Using mock model for demonstration"
        })

@app.route("/api/classify", methods=["POST"])
def classify():
    try:
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
        classification_result, model_info = classify_image(img_array)
        
        return jsonify({
            "success": True,
            "data": classification_result,
            "model_info": model_info
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

# if __name__ == "__main__":
#     print("🧠 BrainAI Diagnosis Platform")
#     print("=" * 40)
#     print(f"Model Status: {'✅ Loaded' if MODEL_LOADED else '❌ Mock Model'}")
#     print(f"Model Format: .keras")
#     if MODEL_LOADED:
#         print(f"Model Path: {MODEL_PATH}")
#         print(f"Input Shape: {loaded_model.input_shape}")
#         print(f"Output Shape: {loaded_model.output_shape}")
#     print("=" * 40)
    
#     app.run(debug=True, port=5173, host='0.0.0.0')
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT environment variable
    app.run(debug=False, port=port, host='0.0.0.0')
