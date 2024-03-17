"""
Module docstring: This module contains the code for preparing/pre-processing the data for training and testing.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

def load_annotations(csv_path):
    """
    Load annotations from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing annotations.

    Returns:
        pd.DataFrame: DataFrame containing the loaded annotations.
    """
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading annotations from {csv_path}: {e}")
        return None

def preprocess_images(df, image_dir, target_size=(128, 128)):
    """
    Preprocess images and labels from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.
        image_dir (str): Directory where images are stored.
        target_size (tuple): Target size for the images as (width, height).

    Returns:
        tuple: A tuple containing:
            - np.array: Preprocessed images as numpy arrays.
            - np.array: Encoded labels as numpy arrays.
            - int: Number of unique classes.
    """
    images, labels = [], []

    # Initialize and fit label encoder
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label_name'])
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    for _, row in df.iterrows():
        try:
            img_path = os.path.join(image_dir, row['image_name'])
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
            labels.append(row['label_encoded'])
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    return np.array(images, dtype='float32'), np.array(labels), num_classes

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Args:
        df (pd.DataFrame): DataFrame to split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: Split data as (train_set, test_set).
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)
