#main.py
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Precision, Recall

from sklearn.model_selection import KFold

# Import other module functions
from data_preparation import create_generators, plot_loss_tf
from data_preprocessing import load_annotations, split_data

num_classes = 3

image_dir = 'data/images_dataset'  # Make sure this is the correct path to your images
csv_path = 'data/labels.csv'  # Make sure this is the correct path to your labels

# Load annotations
annotations = load_annotations(csv_path)

# Split data into training and testing sets
train_df, test_df = split_data(annotations)

# Create data generators
train_images, test_images = create_generators(train_df, test_df, image_dir)

print(train_images.class_indices)
print(test_images.class_indices)

# Compute class weights
labels = train_df['label_name'].values
unique_labels = np.unique(labels)
label_to_index = {label: index for index, label in enumerate(unique_labels)}
train_labels_index = np.array([label_to_index[label] for label in labels])

# After encoding labels but before training
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label_encoded']),  # Ensure this uses the encoded labels
    y=train_df['label_encoded']
)
class_weights_dict = dict(enumerate(class_weights))


# Create the model
def create_model(input_shape=(128, 128, 3), num_classes=3, fine_tune=5):
    inputs = Input(shape=input_shape)
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    # Fine-tuning the top layers of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune]:
        layer.trainable = False

    # Flattening the output of the base model
    flat = Flatten()(base_model.output)

    # Classification head
    class_head = Dense(512, activation='relu')(flat)
    class_head = Dropout(0.3)(class_head)
    class_head = Dense(256, activation='relu')(class_head)
    classes = Dense(num_classes, activation='softmax', name='classes')(class_head)

    model = Model(inputs=inputs, outputs=classes)
    return model

# Learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)

def train_and_evaluate_model(train_data, test_data):
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])

    history = model.fit(
        train_data,
        epochs=30,
        steps_per_epoch=len(train_data),
        validation_data=test_data,
        validation_steps=len(test_data),
        class_weight=class_weights_dict
    )
    return model, history

# Train and evaluate the model
model, history = train_and_evaluate_model(train_images, test_images)

# Plot training and validation loss and accuracy
plot_loss_tf(history)

# Save the best model
model.save('best_model.h5')

print("Model training and first evaluation completed.")
