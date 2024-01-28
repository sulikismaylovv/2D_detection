# data_preparation.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def load_bbox_annotations(csv_path):
    bbox_df = pd.read_csv(csv_path)
    return bbox_df[['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'image_width', 'image_height']]

def encode_labels(bbox_df):
    label_encoder = LabelEncoder()
    bbox_df['label_encoded'] = label_encoder.fit_transform(bbox_df['label_name'])
    return bbox_df, label_encoder.classes_

def process_image_file(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

def prepare_image_data(bbox_df, image_dir, target_size=(128, 128)):
    image_data = []
    boxes = []
    labels = []

    for _, row in bbox_df.iterrows():
        img_array = process_image_file(f"{image_dir}/{row['image_name']}", target_size)
        image_data.append(img_array)
        
        box = [row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']]
        boxes.append(box)
        
        labels.append(row['label_encoded'])
    
    # Convert lists to numpy arrays
    image_data = np.array(image_data, dtype='float32')
    boxes = np.array(boxes, dtype='float32')
    labels = np.array(labels, dtype='int32')
    
    return image_data, boxes, labels

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

    # Process train data
    train_df = train_data[['image_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].copy()
    train_df['bbox'] = train_df.apply(lambda row: [row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']], axis=1)
    
    # Process test data
    test_df = test_data[['image_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].copy()
    test_df['bbox'] = test_df.apply(lambda row: [row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']], axis=1)

    # Create generators
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col='image_name',
        y_col='bbox',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='raw',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col='image_name',
        y_col='bbox',
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='raw',
        batch_size=batch_size,
        shuffle=False
    )

    return train_images, test_images


def split_data(bbox_df, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(bbox_df, test_size=test_size, random_state=random_state)
    return train_data, test_data

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