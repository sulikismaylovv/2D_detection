import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from PIL import Image
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Load CSV
csv_path = 'data\labels.csv'
df = pd.read_csv(csv_path)

# Load images
image_folder = 'data/images_dataset'
images_list = []
y_labels = []

for idx, row in df.iterrows():
    image_filename = row['image_name']
    x, y, width, height = row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']

    # Construct the full path to the image
    image_path = os.path.join(image_folder, image_filename)

    # Load image
    img = np.array(Image.open(image_path))
    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    x /= row['image_width']
    y /= row['image_height']
    width /= row['image_width']
    height /= row['image_height']
    # For simplicity, assuming a single class for each box
    label_info = [x, y, width, height]

    # Append the image and label to the lists
    images_list.append(img)
    y_labels.append(label_info)

# Convert lists to NumPy arrays
X_images = np.array(images_list)
y_labels = np.array(y_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_images, y_labels, test_size=0.2, random_state=42)

# Model Definition
num_classes = 4  # Adjust based on your dataset
epochs = 30

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='linear'))  # Output: [bbox_x, bbox_y, bbox_width, bbox_height]

# Compile the model with mean squared error loss for regression
model.compile(optimizer='adam', loss='mse')

# Model Training
model.fit(X_train, y_train, epochs=epochs, batch_size=32)


# Make predictions on the test set
y_pred = model.predict(X_test)


# Denormalize the bounding box coordinates
y_pred[:, 0] *= X_test.shape[2]  # bbox_x
y_pred[:, 1] *= X_test.shape[1]  # bbox_y
y_pred[:, 2] *= X_test.shape[2]  # bbox_width
y_pred[:, 3] *= X_test.shape[1]  # bbox_height

# Visualize the results

# Visualize the results
for i in range(len(X_test)):
    image = X_test[i]
    true_bbox = y_test[i]
    pred_bbox = y_pred[i]

    # Convert bounding box format from [x, y, width, height] to [x1, y1, x2, y2]
    true_bbox = [true_bbox[0], true_bbox[1], true_bbox[0] + true_bbox[2], true_bbox[1] + true_bbox[3]]
    pred_bbox = [pred_bbox[0], pred_bbox[1], pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]]

    # Visualize the image with true bounding box in blue and predicted bounding box in red
    plt.imshow(image)

    # Plot true bounding box in blue
    plt.gca().add_patch(plt.Rectangle((true_bbox[0] * image.shape[1], true_bbox[1] * image.shape[0]),
                                      true_bbox[2] * image.shape[1] - true_bbox[0] * image.shape[1],
                                      true_bbox[3] * image.shape[0] - true_bbox[1] * image.shape[0],
                                      linewidth=2, edgecolor='b', facecolor='none'))

    # Plot predicted bounding box in red
    plt.gca().add_patch(plt.Rectangle((pred_bbox[0] * image.shape[1], pred_bbox[1] * image.shape[0]),
                                      pred_bbox[2] * image.shape[1] - pred_bbox[0] * image.shape[1],
                                      pred_bbox[3] * image.shape[0] - pred_bbox[1] * image.shape[0],
                                      linewidth=2, edgecolor='r', facecolor='none'))

    plt.show()
