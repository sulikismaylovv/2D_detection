# evaluation.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def plot_bounding_boxes(image, true_boxes, pred_boxes, title=""):
    """
    Plots the actual and predicted bounding boxes on the image.
    
    :param image: The image data
    :param true_boxes: Actual bounding boxes
    :param pred_boxes: Predicted bounding boxes
    :param title: Title of the plot
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot true boxes
    for box in true_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='g', facecolor='none', label='True Box')
        ax.add_patch(rect)

    # Plot predicted boxes
    for box in pred_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none', label='Pred Box')
        ax.add_patch(rect)

    plt.title(title)
    plt.show()

def evaluate_model(model, test_generator, num_samples=10):
    """
    Evaluates the model on a number of samples from the test generator and visualizes the predictions.
    
    :param model: The trained model
    :param test_generator: The test data generator
    :param num_samples: Number of samples to evaluate
    """
    for i in range(num_samples):
        # Retrieve the next batch from the test generator
        x, y_true = next(test_generator)
        y_pred = model.predict(x)

        # Convert the first image in the batch to a displayable format
        image = x[0] * 255  # Assuming preprocessing involves scaling by 1/255
        image = np.array(image, dtype=np.uint8)

        # True and predicted boxes for the first image in the batch
        true_boxes = y_true[0].reshape(-1, 4)
        pred_boxes = y_pred[0].reshape(-1, 4)

        # Plot the image with bounding boxes
        plot_bounding_boxes(image, true_boxes, pred_boxes, title=f"Sample {i+1}")

# Usage in your main script
# from evaluation import evaluate_model
# evaluate_model(model, test_images)
