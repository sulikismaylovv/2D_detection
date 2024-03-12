import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def create_generators(train_data, test_data, image_dir, batch_size=32):
    # ImageDataGenerator for training with augmentation
    train_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
    )

    # ImageDataGenerator for testing without augmentation but with preprocessing
    test_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
    )

    # Encode labels
    label_encoder = LabelEncoder()
    train_data['label_encoded'] = label_encoder.fit_transform(train_data['label_name'])
    test_data['label_encoded'] = label_encoder.transform(test_data['label_name'])

    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # Convert labels to one-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_data['label_encoded'], num_classes=num_classes)
    test_labels = tf.keras.utils.to_categorical(test_data['label_encoded'], num_classes=num_classes)

    # Create generators
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_data,
        directory=image_dir,
        x_col='image_name',
        y_col='label_name',
        target_size=(256, 256),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_data,
        directory=image_dir,
        x_col='image_name',
        y_col='label_name',
        target_size=(256, 256),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size
    )
    
    

    return train_images, test_images


def plot_loss_tf(history):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(history.history['loss'], label='training_loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

import os

def generate_augmented_images_from_generator(train_generator, total_number_per_image, output_directory):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Initialize the ImageDataGenerator
    data_gen = train_generator
    
    # Generate augmented images
    for i, (images, labels) in enumerate(data_gen):
        if i >= len(train_generator.filenames):
            break
        
        image_name = os.path.basename(train_generator.filenames[i])
        image_name_no_extension, _ = os.path.splitext(image_name)
        
        for j in range(total_number_per_image):
            augmented_image_path = os.path.join(output_directory, f"{image_name_no_extension}_{j}.jpg")
            augmented_image = images[j].astype(np.uint8)
            tf.keras.preprocessing.image.save_img(augmented_image_path, augmented_image)

