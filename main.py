"""
Module docstring: This module containes the code to request predictions from the Flask app.
"""
import requests

# The URL of the Flask endpoint for predictions
FLASK_PREDICT_URL = "http://localhost:5001/predict"

# Path to the image you want to predict
img_path = 'test3.jpeg'

# Make a POST request to the Flask app
response = requests.post(FLASK_PREDICT_URL, json={"img_path": img_path})

# Assuming the Flask app returns a JSON with a "result" key
if response.status_code == 200:
    print("Prediction result:", response.json())
else:
    print("Failed to get prediction:", response.text)
