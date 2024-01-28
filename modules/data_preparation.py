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
    test_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
    )

    # One-hot encode the 'label_name' column
    label_encoder = LabelEncoder()
    train_data['label_encoded'] = label_encoder.fit_transform(train_data['label_name'])
    test_data['label_encoded'] = label_encoder.transform(test_data['label_name'])

    # Get the number of classes (this assumes label_encoder has been fitted on all possible labels)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # Convert integer encoded labels to one-hot encoding
    train_labels_one_hot = tf.keras.utils.to_categorical(train_data['label_encoded'], num_classes=num_classes)
    test_labels_one_hot = tf.keras.utils.to_categorical(test_data['label_encoded'], num_classes=num_classes)

    train_data['bbox'] = train_data.apply(lambda row: [row['bbox_x'], row['bbox_y'], row['bbox_x'] + row['bbox_width'], row['bbox_y'] + row['bbox_height']], axis=1)
    test_data['bbox'] = test_data.apply(lambda row: [row['bbox_x'], row['bbox_y'], row['bbox_x'] + row['bbox_width'], row['bbox_y'] + row['bbox_height']], axis=1)

    #print(train_data)
    #print(test_data)
    
    # Add bbox and one-hot encoded labels to train_data and test_data DataFrames
    train_data[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']] = pd.DataFrame(train_data['bbox'].tolist(), index=train_data.index)
    test_data[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']] = pd.DataFrame(test_data['bbox'].tolist(), index=test_data.index)

    for i in range(num_classes):
        train_data[f'class_{i}'] = train_labels_one_hot[:, i]
        test_data[f'class_{i}'] = test_labels_one_hot[:, i]

    # Define 'y_col' as a list of column names to be used as target labels
    y_col = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'] + [f'class_{i}' for i in range(num_classes)]
    #print(train_data)
    #print(test_data)
    #print(y_col)
    # Create generators
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_data,
        directory=image_dir,
        x_col='image_name',
        y_col=y_col,
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
        y_col=y_col,
        target_size=(128, 128),
        color_mode='rgb',
        class_mode='multi_output',
        batch_size=batch_size,
        shuffle=False
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

