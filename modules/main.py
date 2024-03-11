#main.py
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.patches as patches


from pathlib import Path
import os.path
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Precision, Recall, F1Score

from sklearn.model_selection import KFold

# Import other module functions
from data_preparation import create_generators, plot_loss_tf
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

# create train_bboxes and test_bboxes
train_bboxes = train_df['bbox'].values.tolist()
test_bboxes = test_df['bbox'].values.tolist()

# create train_labels and test_labels
train_labels = train_df['label_encoded'].values.tolist()
test_labels = test_df['label_encoded'].values.tolist()
    
    
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
results = best_model.evaluate(test_images)
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

# Save Best Model
best_model.save('best_model.h5')
    
    
