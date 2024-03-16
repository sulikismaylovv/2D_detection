
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# Assuming other modules are refactored as well
from data_preparation import create_image_data_generators, plot_loss_tf, generate_augmented_images
from data_preprocessing import load_annotations, split_data

def create_model(input_shape=(256, 256, 3), num_classes=3):
    """Creates and returns the VGG16-based model for image classification.
    
    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of classes in the dataset.

    Returns:
        A TensorFlow model configured for image classification.
    """
    inputs = Input(shape=input_shape)
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

def train_and_evaluate_model(train_data, test_data, num_classes=3):
    """Trains and evaluates the model using the provided data.
    
    Args:
        train_data: The training data generator.
        test_data: The testing data generator.
        num_classes (int): The number of classes in the dataset.

    Returns:
        The trained model and its training history.
    """
    model = create_model(num_classes=num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_data,
        epochs=30,
        validation_data=test_data,
        callbacks=[early_stopping]
    )
    return model, history

if __name__ == '__main__':
    num_classes = 3
    image_dir = 'data/'  # Updated to more explicitly define image directory
    csv_path = 'data/augmented_labels.csv'  # Updated for clarity
    
    annotations = load_annotations(csv_path)
    # Uncomment this if code is running on LINUX or MacOS
    #annotations['image_name'] = annotations['image_name'].str.replace('\\', '/', regex=False)
    train_df, test_df = split_data(annotations)
    
    preprocess_func = tf.keras.applications.vgg16.preprocess_input
    
    train_images, test_images = create_image_data_generators(preprocess_func, image_dir, train_df, test_df)
    
    # Generate and save augmented images
    #generate_augmented_images(train_images, 'data/output', 5)
    
    # Training and evaluating the model
    model, history = train_and_evaluate_model(train_images, test_images, num_classes=num_classes)
    
    # Plot training and validation loss
    plot_loss_tf(history)
    
    # Save the trained model
    current_time = time.strftime("%d-%H-%M", time.localtime())
    model.save(f'models/model_{current_time}.h5')
    
    print("Model training and evaluation completed.")
