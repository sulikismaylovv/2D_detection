# data_preprocessing.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def load_bbox_annotations(csv_path):
    """
    Load bounding box annotations from a CSV file.
    """
    return pd.read_csv(csv_path)

def plot_image_with_boxes(image_path, boxes, title="Image with Bounding Boxes"):
    """
    Plot an image with bounding boxes.

    :param image_path: Path to the image file.
    :param boxes: List of bounding boxes, each box is a tuple (x, y, width, height).
    :param title: Title of the plot.
    """
    image = plt.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.title(title)
    plt.show()

def preprocess_images_and_boxes(df, image_dir, target_size=(128, 128), num_classes=3):
    """
    Preprocess images and their corresponding bounding boxes and labels.

    :param df: DataFrame containing image paths, bounding boxes, and labels.
    :param image_dir: Directory where images are stored.
    :param target_size: Target size for the images.
    :param num_classes: Total number of classes.
    :return: Preprocessed images, bounding boxes, and labels.
    """
    images = []
    boxes = []
    labels = []

    # Assuming the label column in df is named 'label_name'
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label_name'])
    class_labels = to_categorical(df['label_encoded'], num_classes=num_classes)

    for idx, row in df.iterrows():
        # Image preprocessing
        img_path = os.path.join(image_dir, row['image_name'])
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img_array)

        # Bounding box preprocessing
        x = row['bbox_x'] / row['image_width']
        y = row['bbox_y'] / row['image_height']
        width = row['bbox_width'] / row['image_width']
        height = row['bbox_height'] / row['image_height']
        boxes.append([x, y, width, height])

        # Append class label
        labels.append(class_labels[idx])

    return np.array(images), np.array(boxes), np.array(labels)

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)


