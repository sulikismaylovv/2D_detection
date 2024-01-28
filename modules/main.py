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

def create_rcnn_model(input_shape=(128, 128, 3), num_classes=3):
    inputs = Input(shape=input_shape)
    # Base model, without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False

    # Use a lambda layer to convert feature maps to a single vector
    x = Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2]))(base_model.output)
    
    # Bounding box head
    bbox_head = Dense(256, activation='relu')(x)
    bbox_head = Dropout(0.5)(bbox_head)
    bboxes = Dense(4, activation='sigmoid', name='bboxes')(bbox_head)
    
    # Classification head
    class_head = Dense(256, activation='relu')(x)
    class_head = Dropout(0.5)(class_head)
    classes = Dense(num_classes, activation='softmax', name='classes')(class_head)

    # The model
    model = Model(inputs=inputs, outputs=[bboxes, classes])
    
    return model

model = create_rcnn_model()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss={'bboxes': 'mse', 'classes': 'categorical_crossentropy'},
              metrics={'bboxes': 'mean_squared_error', 'classes': 'accuracy'})

# Displaying model summary
model.summary()

# Training the model using the data generators
history = model.fit(
    train_images,
    epochs=30,
    steps_per_epoch=len(train_images),
    validation_data=test_images,
    validation_steps=len(test_images)
)

# Evaluating the model on the test set using the test data generator
results = model.evaluate(test_images)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plotting training and validation loss
plot_loss_tf(history)

# Predicting bounding box coordinates on the test set using the test data generator
pred_bbox = model.predict(test_images)

# Additional visualization or analysis here
# ...

