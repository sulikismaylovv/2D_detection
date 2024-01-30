#main.py
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


from pathlib import Path
import os.path
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Precision, Recall, F1Score

from sklearn.model_selection import KFold

# Import other module functions
from data_preparation import create_generators, plot_loss_tf
from evaluation import evaluate_model
from data_preprocessing import load_bbox_annotations, preprocess_images_and_boxes , plot_image_with_boxes , split_data

num_classes = 3

image_dir = 'data/images_dataset'  # Make sure this is the correct path to your images
csv_path = 'data/labels.csv'  # Make sure this is the correct path to your labels
# Load annotations
bbox_annotations = load_bbox_annotations(csv_path)

# Preprocess images and boxes
images, boxes, labels = preprocess_images_and_boxes(bbox_annotations, image_dir)

# Split data into training and testing sets
train_df, test_df = split_data(bbox_annotations)
# Create data generators
train_images, test_images = create_generators(train_df, test_df, image_dir)
    
# Show train and test df
print(train_df)
print(test_df)

## pritn space between train and test df
print("\
\n\
\n\
\n\
    ")


# create train_bboxes and test_bboxes
train_bboxes = train_df['bbox'].values.tolist()
test_bboxes = test_df['bbox'].values.tolist()

# create train_labels and test_labels
train_labels = train_df['label_encoded'].values.tolist()
test_labels = test_df['label_encoded'].values.tolist()

# Extract a batch from the training data
train_images_batch, train_labels_batch = next(iter(train_images))

# Inspecting the shape of the batch
print("Shape of train_images_batch:", train_images_batch.shape)
# Display the structure of the first element in the labels list
print("First element in train_labels_batch:", train_labels_batch[0])
print("Type of first element in train_labels_batch:", type(train_labels_batch[0]))

# If the first element is a NumPy array, print its shape
if isinstance(train_labels_batch[0], np.ndarray):
    print("Shape of first element in train_labels_batch:", train_labels_batch[0].shape)

# Inspecting the entire structure of train_labels_batch
print("Structure of train_labels_batch:", [type(label) for label in train_labels_batch])
# Similarly for test data
test_images_batch, test_labels_batch = next(iter(test_images))
print("Shape of test_images_batch:", test_images_batch.shape)
# Display the structure of the first element in the labels list
print("First element in test_labels_batch:", test_labels_batch[0])
print("Type of first element in test_labels_batch:", type(test_labels_batch[0]))

# If the first element is a NumPy array, print its shape
if isinstance(test_labels_batch[0], np.ndarray):
    print("Shape of first element in test_labels_batch:", test_labels_batch[0].shape)

# Inspecting the entire structure of test_labels_batch
print("Structure of test_labels_batch:", [type(label) for label in test_labels_batch])

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

    
# Create the model

def create_rcnn_model(input_shape=(128, 128, 3), num_classes=3, fine_tune=5):
    inputs = Input(shape=input_shape)
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    # Fine-tuning the top layers of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune]:
        layer.trainable = False

    # Flattening the output of the base model
    flat = Flatten()(base_model.output)

    # Classification head
    class_head = Dense(512, activation='relu')(flat)  # Increased the number of units
    class_head = Dropout(0.3)(class_head)  # Adjusted the dropout rate
    class_head = Dense(256, activation='relu')(class_head)
    classes = Dense(num_classes, activation='softmax', name='classes')(class_head)

    model = Model(inputs=inputs, outputs=classes)
    return model


# Learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights))


def train_and_evaluate_model(train_data, test_data):
    model = create_rcnn_model()
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])

    history = model.fit(
        train_data,
        epochs=30,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=len(test_data),
        class_weight=class_weights
    )
    return model, history

# K-Fold Cross Validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
models = []
histories = []

for train, test in kfold.split(images, labels):
    print(f'Training for fold {fold_no} ...')
    model, history = train_and_evaluate_model(train_images, test_images)
    models.append(model)
    histories.append(history)
    fold_no += 1

# Selecting the best model
# Assuming selection based on highest validation accuracy
best_model_index = np.argmax([max(history.history['val_accuracy']) for history in histories])
best_model = models[best_model_index]

# Evaluating the best model (if a separate test set is available)
# Replace `separate_test_images` and `separate_test_labels` with your actual test data
results = model.evaluate(test_images)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plotting training and validation loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import pandas as pd



def predict_random_image(model, image_dir, csv_path, input_shape=(128, 128)):
    # Load the CSV file to get the labels
    bbox_annotations = pd.read_csv(csv_path)

    # List all images in the directory
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    random_image = random.choice(image_files)
    
    # Load and preprocess the image
    img = load_img(os.path.join(image_dir, random_image), target_size=input_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)

    # Retrieve the actual label from the CSV file
    actual_label = bbox_annotations[bbox_annotations['image_name'] == random_image]['label_name'].values[0]

    # Map the predicted class index to its label name
    label_encoder = LabelEncoder()
    bbox_annotations['label_encoded'] = label_encoder.fit_transform(bbox_annotations['label_name'])
    predicted_label = label_encoder.inverse_transform(predicted_class_index)[0]

    # Display the image and prediction
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_label}, Actual Class: {actual_label}")
    plt.show()

# Example usage
for i in range(5):
    predict_random_image(model, image_dir, csv_path)