"""
Utility functions for the web app
"""

import os

from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from config import loaded_model
from flask import jsonify
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


classification_classes = {
    0: 'Pneumonia',
    1: 'Covid',
    2: 'Lung_Opacity',
    3: 'Normal',
    4: 'Tuberculosis'
}

def preprocess_image(image) -> np.array:
    """
    Here the input image is preprocessed for classification

    Parameters: 
    image (PIL.image): This is the input image to be preprocessed

    It returns the preprocessed images as a Numpy array - np.array
    """
    image = Image.open(image).convert("RGB").resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # print
    return image

def classify_image(image: np.array) -> dict:
    
    classification = loaded_model.predict(image, verbose=0)[0]
    classified_label = classification_classes[np.argmax(classification)]
    print(classification)
    print(classification.shape)
    return {
        "classification": classified_label,
        "Pneumonia": round(float(classification[0]), 6),
        "Covid": round(float(classification[1]), 6),
        "Lung_Opacity": round(float(classification[2]), 6),
        "Normal": round(float(classification[3]), 6),
        "Tuberculosis": round(float(classification[4]), 6)
    }

# console.log(classification.shape)
# console.log(classification)
