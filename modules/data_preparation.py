# data_preparation.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def create_generators(train_data, test_data, image_dir, batch_size=32):
    # ImageDataGenerator for training
    train_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    # ImageDataGenerator for testing (validation)
    test_generator = ImageDataGenerator()
    # You should create a copy of train_data and test_data to avoid modifying the original DataFrame
    train_df = train_data.copy()
    test_df = test_data.copy()

    # data_preparation.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def create_generators(train_data, test_data, image_dir, batch_size=32):
    # ImageDataGenerator for training
    train_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

     # ImageDataGenerator for testing (validation)
    test_generator = ImageDataGenerator()

    # Ensure 'bbox' is a list of lists and 'class' is one-hot encoded for both train and test DataFrames
    train_data['bbox'] = train_data.apply(lambda row: [row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']], axis=1)
    test_data['bbox'] = test_data.apply(lambda row: [row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']], axis=1)

    # One-hot encode the 'class' column
    train_data = pd.concat([train_data, pd.get_dummies(train_data['label_name'], prefix='class')], axis=1)
    test_data = pd.concat([test_data, pd.get_dummies(test_data['label_name'], prefix='class')], axis=1)

    # Create generators
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_data,
        directory=image_dir,
        x_col='image_name',
        y_col=['bbox'] + [col for col in train_data.columns if col.startswith('class')],
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='multi_output',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_data,
        directory=image_dir,
        x_col='image_name',
        y_col=['bbox'] + [col for col in test_data.columns if col.startswith('class')],
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='multi_output',
        batch_size=batch_size,
        shuffle=False
    )

    return train_images, test_images

def plot_loss_tf(history):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    widgvis(fig)
    ax.plot(history.history['loss'], label='training_loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()

# Function to visualize widgets
def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

def plot_loss_tf(history):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    widgvis(fig)
    ax.plot(history.history['loss'], label='training_loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()

# Function to visualize widgets
def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False