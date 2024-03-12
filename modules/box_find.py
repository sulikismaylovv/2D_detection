import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load EfficientDet model from TensorFlow Hub
model_url = 'https://tfhub.dev/tensorflow/efficientdet/d4/1'
model = hub.load(model_url)

# Load and preprocess the image
img_path = 'C:/Users/suley/Downloads/boxes.jpeg'
img = Image.open(img_path)
img_array = np.array(img)
input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.uint8)

# Perform inference
detections = model(input_tensor)

# Set a confidence threshold
confidence_threshold = 0.05

# Perform non-maximum suppression
selected_indices = tf.image.non_max_suppression(
    detections['detection_boxes'][0],
    detections['detection_scores'][0],
    max_output_size=200,  # Adjust as needed
    iou_threshold=0.1   # Adjust as needed
)

# Visualize the results
plt.imshow(img_array)

# Draw bounding boxes only for selected indices
for index in selected_indices.numpy():
    box = detections['detection_boxes'][0][index].numpy()
    score = detections['detection_scores'][0][index].numpy()

    if score >= confidence_threshold:
        ymin, xmin, ymax, xmax = box
        ymin, xmin, ymax, xmax = ymin * img_array.shape[0], xmin * img_array.shape[1], ymax * img_array.shape[0], xmax * img_array.shape[1]
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='r', linewidth=2)
        plt.gca().add_patch(rect)
        plt.gca().text(xmin, ymin - 2, f'{score:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        print(f'Confidence: {score:.2f}, Box: {box}')

plt.show()

