# Import needed libraries
import numpy as np
import pandas as pd
import itertools
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import linear, relu, sigmoid
from sklearn.model_selection import ParameterGrid
import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout

# Setting up logging and TensorFlow verbosity
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# Importing OS libraries
from pathlib import Path
import os.path
import os

# Function to walk through directories and count files
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Function to plot loss
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

# Function to calculate model results
def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results

# Function to visualize widgets
def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

# Function to make confusion matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           yticklabels=labels)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

# Load bounding box annotations from the CSV file
def load_bbox_annotations(csv_path):
    bbox_df = pd.read_csv(csv_path)

    # Extract relevant columns
    bbox_annotations = bbox_df[['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'image_width', 'image_height']]

    return bbox_annotations
def count_boxes_and_add_labels(bbox_annotations):
    bbox_count_dict = {}  # Dictionary to store the count of bounding boxes per image
    labeled_boxes = []  # List to store information about each bounding box

    for idx, row in bbox_annotations.iterrows():
        image_filename = row['image_name']

        # Update the count for the current image
        if image_filename in bbox_count_dict:
            bbox_count_dict[image_filename] += 1
        else:
            bbox_count_dict[image_filename] = 1

        x, y, width, height = row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']

        # Assign label based on box size
        if width * height <= 0.01:
            label = 'small_box'
        elif 0.01 < width * height <= 0.05:
            label = 'medium_box'
        else:
            label = 'large_box'

        # For simplicity, assuming a single class for each box
        label_info = [x, y, width, height, label]

        labeled_boxes.append({
            'image_name': image_filename,
            'bbox_info': label_info
        })

    return bbox_count_dict, labeled_boxes
# Function to create the VGG16-based model
def create_vgg16_model():
    # Load the pretrained VGG16 model
    pretrained_model = VGG16(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False

    # Add custom layers on top of the pretrained model
    model = Sequential([
        layers.experimental.preprocessing.Resizing(128, 128),
        layers.experimental.preprocessing.Rescaling(1. / 255),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(4, activation='sigmoid')  # Assuming 4 output units for bounding box coordinates (adjust as needed)
    ])

    return model

# Set the current working directory
print("Current Working Directory:", os.getcwd())
dataset = 'data/images_dataset'

if os.path.exists(dataset):
    print("Path exists:", dataset)
else:
    print("Path does not exist:", dataset)



# Walk through the dataset directory
walk_through_dir(dataset)

# Load bounding box annotations from CSV
csv_path = 'data/labels.csv'
bbox_annotations = load_bbox_annotations(csv_path)

# Display information about 11 images
num_images = 11
fig, axes = plt.subplots(4, 3, figsize=(10, 10))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

widgvis(fig)
for i, ax in enumerate(axes.flat):
    if i >= len(bbox_annotations['image_name']):
        break

    img_path = os.path.join(dataset, bbox_annotations['image_name'].iloc[i])
    img = plt.imread(img_path)
    ax.imshow(img)

    print(f"Image {i+1} Path: {img_path}")

    # Draw bounding boxes on the image
    for _, row in bbox_annotations[bbox_annotations['image_name'] == bbox_annotations['image_name'].iloc[i]].iterrows():
        rect = plt.Rectangle((row['bbox_x'], row['bbox_y']), row['bbox_width'], row['bbox_height'], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.set_title(f"Image: {bbox_annotations['image_name'].iloc[i]}")
    ax.set_axis_off()

fig.suptitle("Bounding Boxes on Images", fontsize=14)
plt.show()

# Convert labels to numerical format
label_encoder = LabelEncoder()
bbox_annotations['label_encoded'] = label_encoder.fit_transform(bbox_annotations['label_name'])

# Group bounding box annotations by image and aggregate bounding boxes and labels into lists
grouped_annotations = bbox_annotations.groupby('image_name').agg(
    {'bbox_x': list, 'bbox_y': list, 'bbox_width': list, 'bbox_height': list, 'label_encoded': list}
).reset_index()


bbox_count_dict, labeled_boxes = count_boxes_and_add_labels(bbox_annotations)

# Display the count of bounding boxes per image and their labels ...
for image_filename, count in bbox_count_dict.items():
    print(f"{image_filename}: {count} bounding boxes")
    labels_for_image = [box['bbox_info'][4] for box in labeled_boxes if box['image_name'] == image_filename]
    print(f"Labels for {image_filename}: {labels_for_image}")

# Split the data into training and testing sets
train_data, test_data = train_test_split(grouped_annotations, test_size=0.2, random_state=42)


# Create train image generator
# Convert lists to arrays for training data
train_data['bbox_x'] = train_data['bbox_x'].apply(np.array)
train_data['bbox_y'] = train_data['bbox_y'].apply(np.array)
train_data['bbox_width'] = train_data['bbox_width'].apply(np.array)
train_data['bbox_height'] = train_data['bbox_height'].apply(np.array)
train_data['label_encoded'] = train_data['label_encoded'].apply(np.array)

# Convert lists to arrays for test data
test_data['bbox_x'] = test_data['bbox_x'].apply(np.array)
test_data['bbox_y'] = test_data['bbox_y'].apply(np.array)
test_data['bbox_width'] = test_data['bbox_width'].apply(np.array)
test_data['bbox_height'] = test_data['bbox_height'].apply(np.array)
test_data['label_encoded'] = test_data['label_encoded'].apply(np.array)

# Create ImageDataGenerator for training and testing images
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

# Print filenames from CSV
print("Filenames from CSV:")
print(list(train_data['image_name']))
dataset = 'data/images_dataset'
# Create train image generator
train_images = train_generator.flow_from_dataframe(
    dataframe=train_data,
    directory=dataset,
    x_col='image_name',
    y_col=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'label_encoded'],
    target_size=(128, 128),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42
)

# Print filenames generated by the generator
print("Filenames generated by the generator:")
print(train_images.filenames)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_data,
    directory=dataset,
    x_col='image_name',
    y_col=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'label_encoded'],
    target_size=(128, 128),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

# Create the VGG16-based model
model = create_vgg16_model()
# Build the model by calling it on a sample input
sample_batch = next(iter(train_images))  # Get a sample batch from the training data
model.build(input_shape=sample_batch[0].shape)
# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',  # Use Mean Squared Error as the loss for bounding box regression
    metrics=['accuracy']
)

# Display model summary
model.summary()
# Print filenames from CSV
print("Filenames from CSV:")
print(list(train_data['image_name']))
print(len(train_data))

# Print actual filenames in the dataset directory
print("Actual Filenames in Dataset Directory:")
dataset_path = 'data/images_dataset'
actual_filenames = [filename for filename in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, filename))]
print(actual_filenames)

# For training data
train_images_batch, train_labels_batch = next(iter(train_images))
print("Shape of train_images_batch:", train_images_batch.shape)  # (12, 128, 128, 3)
print("Shape of train bounding boxes:", train_labels_batch[0].shape)  # (12, 4)
print("Shape of train class labels:", train_labels_batch[1].shape)  # (12, num_classes)

# For testing data
test_images_batch, test_labels_batch = next(iter(test_images))
print("Shape of test_images_batch:", test_images_batch.shape)  # (4, 128, 128, 3)
print("Shape of test bounding boxes:", test_labels_batch[0].shape)  # (4, 4)
print("Shape of test class labels:", test_labels_batch[1].shape)  # (4, num_classes)

# Train the model
history = model.fit(
    train_images,
    steps_per_epoch=len(train_images),
    epochs=30
)

# Evaluate the model on the test set
results = model.evaluate(test_images, verbose=0)
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])

# Plot training loss
plot_loss_tf(history)
print("aarived at preditc")
# Predict bounding box coordinates on the test set
pred_bbox = model.predict(test_images)

# Print all the image paths in test_images
for i in range(len(test_images)):
    batch = test_images[i]
    image_paths = batch[0]

    for image_path in image_paths:
        print("Image Path:", image_path)