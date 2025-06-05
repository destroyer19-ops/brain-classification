"""
Create Flask App, load model and define CORS
"""
import os
from flask import Flask
from flask_cors import CORS
from tensorflow.keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "./model/MobileNet_Parkinson_diagnosis.keras")
try:
    loaded_model = load_model(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    loaded_model = None
    MODEL_LOADED = False
