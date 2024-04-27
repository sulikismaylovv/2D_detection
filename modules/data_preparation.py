"""
Module docstring: This module contains the code for preparing the data for training and testing.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

def create_image_data_generators(preprocess_input_func, image_dir, train_df, test_df, batch_size=32, target_size=(256, 256)):
    """
    Creates image data generators for training and testing.
    
    Args:
        preprocess_input_func: Preprocessing function to apply to each image.
        image_dir: Directory where images are located.
        train_df: DataFrame containing training data.
        test_df: DataFrame containing test data.
        batch_size: Size of the batches of data.
        target_size: Tuple of integers, the dimensions to which all images found will be resized.

    Returns:
        A tuple of (train_generator, test_generator).
    """
    # Training ImageDataGenerator with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Testing ImageDataGenerator without augmentation, only preprocessing
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_func
    )
    
    # Creating generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col='image_name',
        y_col='label_name',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col='image_name',
        y_col='label_name',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator

def encode_labels(train_df, test_df):
    """
    Encodes the labels in the training and testing DataFrame using LabelEncoder and adds the encoded labels as a new column.
    
    Args:
        train_df: DataFrame containing training data.
        test_df: DataFrame containing test data.

    Returns:
        A tuple of (train_df, test_df) with the encoded labels added.
    """
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label_name'])
    test_df['label_encoded'] = label_encoder.transform(test_df['label_name'])
    
    return train_df, test_df

def plot_loss_tf(history):
    """
    Plots the training and validation loss.

    Args:
        history: A TensorFlow History object from model.fit().
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def create_test_generator(test_df, image_dir, batch_size=32, target_size=(256, 256)):
    """Creates a data generator for data augmentation for the given test DataFrame.

    Args:
        test_df: DataFrame containing test data.
        image_dir: Directory where images are located.
        batch_size: Size of the batches of data.
        target_size: Tuple of integers, the dimensions to which all images found will be resized.

    Returns:
        A data generator for the test data.
    """
    # Initialize the data generator with rescale to ensure the image is properly normalized
    datagen = ImageDataGenerator(rescale=1./255)

    # Create the generator to read images found in dataframe
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col="image_name",
        y_col="label_name",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # choose 'binary' if you have binary classification
        shuffle=False  # No need to shuffle the test data
    )

    return test_generator

def generate_augmented_images(generator, output_dir, total_imgs_per_file):
    """
    Generates and saves augmented images using a given generator.

    Args:
        generator: The ImageDataGenerator's flow_from_dataframe method. (Use create_test_generator to create the generator.)
        output_dir: Directory where augmented images will be saved.
        total_imgs_per_file: Number of augmented images to generate per file.
    """
    print(f"Generating augmented images to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over the generator indefinitely
    img_index = 0
    for images, labels in generator:
        # Generate a batch of images
        for j in range(images.shape[0]):
            image = images[j].astype(np.uint8)
            label = labels[j]
            class_label = label.argmax()  # Assuming labels are one-hot encoded

            # Construct an image name with the class label and image index
            augmented_image_name = f"aug_{class_label}_{img_index}.jpg"
            img_index += 1

            # Save the image
            augmented_image_path = os.path.join(output_dir, augmented_image_name)
            tf.keras.preprocessing.image.save_img(augmented_image_path, image)

        # Break after generating the specified number of images
        if img_index >= total_imgs_per_file * len(generator.filenames):
            break



# Example usage:
# if __name__ == "__main__":
#     train_df = pd.read_csv('train.csv')  # Adjust path as necessary
#     test_df = pd.read_csv('test.csv')    # Adjust path as necessary
#     image_dir = 'path/to/images'
#     
#     preprocess_input_func = tf.keras.applications.vgg16.preprocess_input
#     train_generator, test_generator = create_image_data_generators(
#         preprocess_input_func, image_dir, train_df, test_df
#     )
