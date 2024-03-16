
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_model(model_url):
    return hub.load(model_url)

def preprocess_image(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    return tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.uint8)

def perform_inference(model, input_tensor, confidence_threshold=0.05, max_output_size=200, iou_threshold=0.1):
    detections = model(input_tensor)
    selected_indices = tf.image.non_max_suppression(
        detections['detection_boxes'][0],
        detections['detection_scores'][0],
        max_output_size=max_output_size,
        iou_threshold=iou_threshold
    )
    return detections, selected_indices

def visualize_results(img_array, detections, selected_indices, confidence_threshold):
    """
    Visualizes the detection results on the input image.

    Args:
        img_array (numpy.ndarray): The input image array.
        detections (dict): The detection results dictionary.
        selected_indices (numpy.ndarray): The indices of the selected detections.
        confidence_threshold (float): The confidence threshold for displaying a detection.

    Returns:
        None
    """
    plt.imshow(img_array)
    for index in selected_indices.numpy():
        box = detections['detection_boxes'][0][index].numpy()
        score = detections['detection_scores'][0][index].numpy()
        if score >= confidence_threshold:
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = ymin * img_array.shape[0], xmin * img_array.shape[1], ymax * img_array.shape[0], xmax * img_array.shape[1]
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='r', linewidth=2)
            plt.gca().add_patch(rect)
            plt.gca().text(xmin, ymin - 2, f'{score:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.show()

if __name__ == '__main__':
    # Configuration
    config = {
        'model_url': 'https://tfhub.dev/tensorflow/efficientdet/d4/1',
        'img_path': 'test.jpeg',
        'confidence_threshold': 0.05,
        'max_output_size': 200,
        'iou_threshold': 0.1
    }

    # Process
    model = load_model(config['model_url'])
    input_tensor = preprocess_image(config['img_path'])
    detections, selected_indices = perform_inference(model, input_tensor, config['confidence_threshold'], config['max_output_size'], config['iou_threshold'])
    visualize_results(np.squeeze(input_tensor.numpy()), detections, selected_indices, config['confidence_threshold'])
