import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Pre-load models and preprocess label data
det_model_url = 'models/detection'
det_model = tf.saved_model.load(det_model_url)
class_model_path = 'models/model_1710271526.733847.h5'
class_model = tf.keras.models.load_model(class_model_path)

# Preprocess CSV for labels
bbox_annotations = pd.read_csv('data/labels.csv')
label_encoder = LabelEncoder()
bbox_annotations['label_encoded'] = label_encoder.fit_transform(bbox_annotations['label_name'])
encoded_labels = label_encoder.classes_

def predict_image(model, img_array, input_shape=(256, 256)):
    img = Image.fromarray(img_array.astype('uint8'), 'RGB')
    img = img.resize(input_shape)
    img_array_resized = np.array(img)
    img_array_expanded = np.expand_dims(img_array_resized, axis=0)
    prediction = model.predict(img_array_expanded)
    return prediction

def interpret_prediction(prediction):
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = encoded_labels[predicted_class_index[0]]
    return predicted_label

# Process image and detections
img_path = 'test3.jpeg'
img = Image.open(img_path)
img_array = np.array(img)
input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.uint8)

det_detections = det_model(input_tensor)
confidence_threshold = 0.05

selected_indices = tf.image.non_max_suppression(
    det_detections['detection_boxes'][0],
    det_detections['detection_scores'][0],
    max_output_size=100,
    iou_threshold=0.1
)

results = []

for index in selected_indices.numpy():
    box = det_detections['detection_boxes'][0][index].numpy()
    score = det_detections['detection_scores'][0][index].numpy()

    if score >= confidence_threshold:
        ymin, xmin, ymax, xmax = [int(b) for b in box]
        crop_img_array = img_array[ymin:ymax, xmin:xmax]
        prediction = predict_image(class_model, crop_img_array)
        predicted_label = interpret_prediction(prediction)
        
        results.append((predicted_label, score, (xmin, ymin, xmax, ymax)))

for result in results:
    print(f"Label: {result[0]}, Confidence: {result[1]:.2f}, Box: {result[2]}")
