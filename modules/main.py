#main.py
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten



from pathlib import Path
import os.path
import os


# Import other module functions
from data_preparation import load_bbox_annotations, create_generators, plot_loss_tf
from evaluation import evaluate_model
from data_preprocessing import load_bbox_annotations, preprocess_images_and_boxes , plot_image_with_boxes , split_data

image_dir = 'data/images_dataset'  # Make sure this is the correct path to your images
csv_path = 'data/labels.csv'  # Make sure this is the correct path to your labels
# Load annotations
bbox_annotations = load_bbox_annotations(csv_path)

# Preprocess images and boxes
images, boxes = preprocess_images_and_boxes(bbox_annotations, image_dir)

# Split data into training and testing sets
train_df, test_df = split_data(bbox_annotations)
# Create data generators
train_images, test_images = create_generators(train_df, test_df, image_dir)
    
    
    
    

# Function to create the VGG16-based model
def create_vgg16_model(input_shape=(128, 128, 3)):
    pretrained_model = VGG16(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False

    # Add custom layers on top of the pretrained model
    model = Sequential([
        pretrained_model,
        Flatten(),  # Flatten the output of VGG16
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(4, activation='sigmoid')  # Output layer for bounding box coordinates
    ])

    return model

model = create_vgg16_model()
sample_batch = next(iter(train_images))  # Get a sample batch from the training data
sample_batch = next(iter(train_images))
print("Batch image shape:", sample_batch[0].shape)
model.build(input_shape=sample_batch[0].shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])  # Using Mean Absolute Error as a metric

# Display model summary
model.summary()

# Train the model
history = model.fit(train_images, epochs=30, steps_per_epoch=len(train_images), validation_data=test_images, validation_steps=len(test_images))

# Evaluate the model on the test set
results = model.evaluate(test_images)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plot training and validation loss
plot_loss_tf(history)

evaluate_model(model, test_images, num_samples=10)

# Predict bounding box coordinates on the test set
pred_bbox = model.predict(test_images)

# You can also add any additional visualization or analysis as needed here
# ...

