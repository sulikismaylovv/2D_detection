# Image Processing and Recognition Unit for the ORL System

This repository houses a 2D object detection system that utilizes a combination of two models: a pre-trained EfficientNet for object localization and a custom-trained VGG-16 for object recognition. The system can be executed locally or through a Flask API to perform detection on images.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the Image Processing and Recognition Unit, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have Python 3.11 installed.
3. Install the required dependencies by running `pip install -r requirements.txt` from the root of the project.

## Usage

There are two main ways to use this system:

### Local Execution

To run the detection models locally, follow these steps:

1. Execute `combined.py` to run the models. You can do this by opening a command prompt in the project directory and running the following command:
2. Run the command `python combined.py` in the terminal.

### Flask API Execution

To run the detection models through a Flask API, follow these steps:

1. Start the Flask server by running `python app.py` in the terminal.
2. Send a POST request to the `/detect` endpoint with the image data or execute `python main.py` in separate terminal.
3. You ca also use `curl` : 

    `curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d "{"img_path": "test3.jpeg"}" `

Replace `test3.jpeg` with the path to your image file.

## Project Structure

- **data/**: Contains datasets and data-related scripts.
- **models/detection/**: Houses the model files and weights.
- **modules/**: Includes scripts for training the EfficientNet and VGG-16 models.
- **app.py**: The Flask application for running the models as an API.
- **combined.py**: Script to run both models sequentially for local execution.
- **main.py**: Utility script to send requests to the Flask server for predictions.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

To contribute:

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE). See `LICENSE` file for details.

