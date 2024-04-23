"""
Module docstring: This module combines the detection using OpenCV for contour detection and TensorFlow for classification to perform inference on an image.
"""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Assuming the model and history are saved during the training process
def load_best_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(model, img_array, input_shape=(256, 256)):
    img = Image.fromarray(img_array.astype('uint8'), 'RGB')
    img = img.resize(input_shape)
    img_array_resized = np.array(img)
    img_array_expanded = np.expand_dims(img_array_resized, axis=0)  # Add batch dimension
    prediction = model.predict(img_array_expanded)
    return prediction

def interpret_prediction(prediction):
    label_encoder = LabelEncoder()
    bbox_annotations = pd.read_csv('data/labels.csv')  # Replace with your CSV path
    bbox_annotations['label_encoded'] = label_encoder.fit_transform(bbox_annotations['label_name'])
    encoded_labels = label_encoder.classes_

    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = encoded_labels[predicted_class_index[0]]
    return predicted_label

# Load classification model
class_model_path = 'models/model_2204-19-20.keras'  # Replace with your model path
class_model = load_best_model(class_model_path)

# Load and preprocess the image using OpenCV
img_path = 'testV2/test11.jpg'  # Replace with your image path
img = cv2.imread(img_path)
# convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 45, 194)
# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 3, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
thresh = cv2.dilate(thresh,None,iterations =5)
thresh = cv2.erode(thresh,None,iterations =5)

# Find the contours
contours,hierarchy = cv2.findContours(thresh,
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

# Convert back to color image for drawing
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process each contour as a detection
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if 75000 < area < 200000:
        print(f"Detected bounding box: x={x}, y={y}, w={w}, h={h}")
        print(f"Detected area: {area}")
        print(f"Detected Dimensions(length x width): {w} x {h}")
        crop_img_array = img_color[y:y+h, x:x+w]
        prediction = predict_image(class_model, crop_img_array)
        predicted_label = interpret_prediction(prediction)

        # Print raw prediction output and the predicted label
        print(f"Raw prediction output (probabilities for each class): {prediction}")
        print(f"Predicted label: {predicted_label}")

        # Draw the bounding box and label on the image
        rect = Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=2)
        plt.gca().add_patch(rect)
        plt.gca().text(x, y, f'{predicted_label}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

# Show the original image with bounding boxes and recognition labels
plt.imshow(img_color)
plt.axis('off')
plt.show()
