# evaluation.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.metrics import CategoricalAccuracy, MeanAbsoluteError


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
    mae = MeanAbsoluteError()
    cat_acc = CategoricalAccuracy()

    for i in range(num_samples):
        x, y = next(test_generator)
        y_pred = model.predict(x)
        mae.update_state(y['bbox'], y_pred['bbox'])
        cat_acc.update_state(y['class'], y_pred['class'])

    print(f"Bounding Box MAE: {mae.result().numpy()}, Classification Accuracy: {cat_acc.result().numpy()}")


# Usage in your main script
# from evaluation import evaluate_model
# evaluate_model(model, test_images)
