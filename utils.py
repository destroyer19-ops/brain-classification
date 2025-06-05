"""
Utility functions for the web app
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from config import loaded_model, MODEL_LOADED

classification_classes = {
    0: "Alzheimer's Disease",
    1: "Normal",
    2: "Parkinson's Disease",
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
        
        # Get model input shape
        if hasattr(loaded_model, 'input_shape') and loaded_model is not None:
            input_shape = loaded_model.input_shape
            if len(input_shape) == 4:  # (batch, height, width, channels)
                target_size = (input_shape[1], input_shape[2])
            else:
                target_size = (224, 224)
        else:
            target_size = (224, 224)
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to array
        image = img_to_array(image)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Normalize pixel values
        image = image / 255.0
        
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
    if not MODEL_LOADED or loaded_model is None:
        raise ValueError("Model not loaded")
    
    try:
        # Make prediction
        prediction = loaded_model.predict(image_array, verbose=0)[0]
        
        # Validate output shape
        if len(classification_classes) != len(prediction):
            raise ValueError(f"Model output classes ({len(prediction)}) do not match expected classes ({len(classification_classes)})")
        
        # Get the predicted class
        predicted_class_idx = np.argmax(prediction)
        classified_label = classification_classes[predicted_class_idx]
        confidence = float(np.max(prediction))
        
        # Ensure probabilities sum to 1
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
