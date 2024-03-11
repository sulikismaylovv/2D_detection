# Description: This module contains functions to load and preprocess data.
# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

def load_annotations(csv_path):
    """
    Load annotations from a CSV file.
    """
    return pd.read_csv(csv_path)

def preprocess_images(df, image_dir, target_size=(128, 128)):
    """
    Preprocess images and labels.

    :param df: DataFrame containing image paths and labels.
    :param image_dir: Directory where images are stored.
    :param target_size: Target size for the images.
    :return: Preprocessed images and labels.
    """
    images = []
    labels = []

    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label_name'])
    num_classes = len(label_encoder.classes_)
    
    # Print Info
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    for idx, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_name'])
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img_array)

        labels.append(df['label_encoded'][idx])

    return np.array(images), np.array(labels), num_classes

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)
