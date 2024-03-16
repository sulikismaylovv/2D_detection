
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import random
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from data_preparation import create_test_generator

def load_best_model(model_path):
    """
    Loads and returns the best performing model from a given path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        Loaded TensorFlow model.
    """
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def evaluate_model(model, test_images):
    """
    Evaluates the model on the test dataset and prints the results.

    Args:
        model: The TensorFlow model to evaluate.
        test_images: The test data generator.

    """
    results = model.evaluate(test_images)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

def plot_history(history):
    """
    Plots the training and validation accuracy and loss values from the model's history.

    Args:
        history: The history object returned from model.fit().
    """
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Test')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Test')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()

def predict_random_image(model, image_dir, csv_path, input_shape=(256, 256)):
    """
    Predicts a random image from the given directory and prints the prediction.

    Args:
        model: The trained TensorFlow model for prediction.
        image_dir (str): Directory containing the images.
        csv_path (str): Path to the CSV file with image annotations.
        input_shape (tuple): Shape to resize images to before prediction.
    """
    annotations = pd.read_csv(csv_path)
    random_image = random.choice(annotations['image_name'].unique())
    img_path = os.path.join(image_dir, random_image)

    img = load_img(img_path, target_size=input_shape)
    img_array = img_to_array(img) / 255.0
    img_array_expanded = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array_expanded)
    print("Raw prediction output:", prediction)

    # Decode prediction
    label_encoder = LabelEncoder()
    annotations['label_encoded'] = label_encoder.fit_transform(annotations['label_name'])
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    print(f"Predicted class: {predicted_class}")
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

# Example usage
# change model_path to appropriate path
model_path = 'models/best_model.h5'
image_dir = 'data/test_images'
csv_path = 'data/test_annotations.csv'

model = load_best_model(model_path)
if model:
    test_images = create_test_generator(pd.read_csv(csv_path), image_dir)
    evaluate_model(model, test_images)
    plot_history(model.history)  # Assuming history is accessible or loaded similarly
    predict_random_image(model, image_dir, csv_path)
