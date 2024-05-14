"""
Module docstring: This module contains the code for the Flask app that serves the combined model.
"""

from flask import Flask, request, jsonify
import time
from lazy_loader import ModelPipeline
import os

def create_app():
    app = Flask(__name__)

    # Explicitly print server startup steps
    print("Initializing Flask app...")

    print("Loading models...")
    start_time = time.time()  # Start timer
    
    # Get Latest model path from the models folder
    model_path = 'models'
    model_name = os.listdir(model_path)[-1]
    model_path = os.path.join(model_path, model_name)
    

    # Initialize your model pipeline here
    pipeline = ModelPipeline('models/detection', model_path, 'data/labels.csv')

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Models loaded in {elapsed_time:.2f} seconds.")

    @app.route('/health', methods=['GET'])
    def health():
        # Health check endpoint to quickly test if the server is up
        return jsonify({"status": "healthy"}), 200

    @app.route('/predict', methods=['POST'])
    def predict():
        # Prediction endpoint
        try:
            data = request.json
            img_path = data['img_path']
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image path does not exist: {img_path}")
            result = pipeline.predict_and_interpret(img_path)
            print(f"Prediction result: {result}")
            return jsonify({"result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    @app.route('/predict_box', methods=['POST'])
    def predict_box():
        # Prediction endpoint
        try:
            data = request.json
            img_path = data['img_path']
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image path does not exist: {img_path}")
            result = pipeline.predict_and_interpret_classification(img_path)
            print(f"Prediction result: {result}")
            return jsonify({"result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print("Flask app ready to serve requests.")
    return app

    
        

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5001)
