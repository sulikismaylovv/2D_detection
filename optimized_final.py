import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class LazyModelLoader:
    def __init__(self, det_model_path, class_model_path):
        self._det_model = None
        self._class_model = None
        self._det_model_path = det_model_path
        self._class_model_path = class_model_path

    @property
    def det_model(self):
        if self._det_model is None:
            print("Loading EfficientDet model...")
            self._det_model = tf.saved_model.load(self._det_model_path)
        return self._det_model

    @property
    def class_model(self):
        if self._class_model is None:
            print("Loading custom VGG16 model...")
            self._class_model = tf.keras.models.load_model(self._class_model_path)
        return self._class_model

class ModelPipeline:
    def __init__(self, det_model_path, class_model_path, label_csv_path):
        self.model_loader = LazyModelLoader(det_model_path, class_model_path)
        self.labels = self._load_and_encode_labels(label_csv_path)

    def _load_and_encode_labels(self, csv_path):
        bbox_annotations = pd.read_csv(csv_path)
        label_encoder = LabelEncoder()
        label_encoder.fit(bbox_annotations['label_name'])
        return label_encoder

    def predict_and_interpret(self, img_path, input_shape=(256, 256), confidence_threshold=0.05):
        # Load and preprocess image
        img = Image.open(img_path)
        img_array = np.array(img)
        input_tensor = tf.convert_to_tensor([img_array], dtype=tf.uint8)
        
        # Perform detection
        det_detections = self.model_loader.det_model(input_tensor)
        
        # Perform non-maximum suppression
        selected_indices = tf.image.non_max_suppression(
            det_detections['detection_boxes'][0],
            det_detections['detection_scores'][0],
            max_output_size=100,
            iou_threshold=0.1
        )
        
        # Process each detection
        for index in selected_indices.numpy():
            box = det_detections['detection_boxes'][0][index].numpy()
            score = det_detections['detection_scores'][0][index].numpy()
            if score < confidence_threshold:
                continue
            
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin * img_array.shape[0]), int(xmin * img_array.shape[1]), \
                                     int(ymax * img_array.shape[0]), int(xmax * img_array.shape[1])
            
            # Crop and predict
            crop_img_array = img_array[ymin:ymax, xmin:xmax]
            prediction = self._predict_image(crop_img_array, input_shape)
            
            predicted_label = self._interpret_prediction(prediction)
            print(f'Box: {box}, Confidence: {score:.2f}, Label: {predicted_label}')
    
    def _predict_image(self, img_array, input_shape):
        img = Image.fromarray(img_array)
        img = img.resize(input_shape)
        img_array_resized = np.array(img)
        img_array_expanded = np.expand_dims(img_array_resized, axis=0)
        prediction = self.model_loader.class_model.predict(img_array_expanded)
        return prediction
    
    def _interpret_prediction(self, prediction):
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        return self.labels.inverse_transform([predicted_class_index])[0]

# Example usage
det_model_path = 'models/detection'
class_model_path = 'models/model_1710271526.733847.h5'
label_csv_path = 'data/labels.csv'
img_path = 'test3.jpeg'

pipeline = ModelPipeline(det_model_path, class_model_path, label_csv_path)
pipeline.predict_and_interpret(img_path)
