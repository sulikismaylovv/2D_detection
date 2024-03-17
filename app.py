from flask import Flask, request, jsonify
import time
from optimized_final import ModelPipeline
import os

def create_app():
    app = Flask(__name__)

    # Explicitly print server startup steps
    print("Initializing Flask app...")

    print("Loading models...")
    start_time = time.time()  # Start timer

    # Initialize your model pipeline here
    pipeline = ModelPipeline('models/detection', 'models/model_1710271526.733847.h5', 'data/labels.csv')

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

    print("Flask app ready to serve requests.")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5001)