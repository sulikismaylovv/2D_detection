"""
Module docstring: This module combines the detection and classification models to perform inference on an image.
"""

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder


# Assuming the model and history are saved during the training process
def load_best_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(model, img_array, input_shape=(256, 256)):
    # Assuming the input_shape is required by the second model for its input
    img = Image.fromarray(img_array.astype('uint8'), 'RGB')
    img = img.resize(input_shape)
    img_array_resized = np.array(img)
    img_array_expanded = np.expand_dims(img_array_resized, axis=0)  # Add batch dimension
    prediction = model.predict(img_array_expanded)
    return prediction

def interpret_prediction(prediction):
    # Convert the prediction probabilities to class labels
    label_encoder = LabelEncoder()
    bbox_annotations = pd.read_csv('data/labels.csv')  # Replace with your CSV path
    bbox_annotations['label_encoded'] = label_encoder.fit_transform(bbox_annotations['label_name'])
    encoded_labels = label_encoder.classes_
    
    #Print the encoded labels
    print(f"Encoded labels: {encoded_labels}")

    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = encoded_labels[predicted_class_index[0]]
    return predicted_label

# Load EfficientDet model from TensorFlow Hub
det_model_url = 'models/detection'
det_model = tf.saved_model.load(det_model_url)

# Load classification model
class_model_path = 'models/model_2204-19-20.keras'  # Replace with your model path
class_model = load_best_model(class_model_path)

# Load and preprocess the image
img_path = 'testV2/test15.jpg'  # Replace with your image path
img = Image.open(img_path)
img_array = np.array(img)
input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.uint8)

# Perform inference with EfficientDet
det_detections = det_model(input_tensor)

# Set a confidence threshold
confidence_threshold = 0.05

# Perform non-maximum suppression
selected_indices = tf.image.non_max_suppression(
    det_detections['detection_boxes'][0],
    det_detections['detection_scores'][0],
    max_output_size=50,  # Adjust as needed
    iou_threshold=0.01   # Adjust as needed
)

# Process detections and classify each box
for index in selected_indices.numpy():
    box = det_detections['detection_boxes'][0][index].numpy()
    score = det_detections['detection_scores'][0][index].numpy()

    if score >= confidence_threshold:
        ymin, xmin, ymax, xmax = box
        #Print the confidence score and box coordinates
        print(f'Confidence: {score:.2f}, Box: {box}')
        # Add padding around the detected box
        padding = -0.5 # Adjust padding as needed
        ymin, xmin, ymax, xmax = int(ymin * img_array.shape[0]), int(xmin * img_array.shape[1]), \
                                 int(ymax * img_array.shape[0]), int(xmax * img_array.shape[1])
        # Crop the box from the original image
        crop_img_array = img_array[ymin:ymax, xmin:xmax]
        # Predict the content of the box using the classification model
        prediction = predict_image(class_model, crop_img_array)
        # You will need to implement how to interpret the prediction into a class label
        predicted_label = interpret_prediction(prediction)  # Implement this function based on your model output
        
        # Print raw prediction output and the predicted label
        print(f"Raw prediction output (probabilities for each class): {prediction}")
        print(f"Predicted label: {predicted_label}")

        # Draw the bounding box and label on the image
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='r', linewidth=2)
        plt.gca().add_patch(rect)
        plt.gca().text(xmin, ymax, f'{predicted_label} ({score:.2f})', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

# Show the original image with bounding boxes and recognition labels
plt.imshow(img_array)
plt.axis('off')
plt.show()
