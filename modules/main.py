#main.py
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


from pathlib import Path
import os.path
import os


# Import other module functions
from data_preparation import create_generators, plot_loss_tf
from evaluation import evaluate_model
from data_preprocessing import load_bbox_annotations, preprocess_images_and_boxes , plot_image_with_boxes , split_data

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

# create train_bboxes and test_bboxes
train_bboxes = train_df['bbox'].values.tolist()
test_bboxes = test_df['bbox'].values.tolist()

# create train_labels and test_labels
train_labels = train_df['label_encoded'].values.tolist()
test_labels = test_df['label_encoded'].values.tolist()

print(train_images)
print(test_images)


    
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

