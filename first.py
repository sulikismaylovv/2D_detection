import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

csv_path = 'data\labels.csv'
df = pd.read_csv(csv_path)

image_folder = 'data\images_dataset'

# Initialize empty lists to store images and labels
images_list = []  # Renamed from X_images to avoid conflict
y_labels = []

# Iterate through the rows in the CSV file
for idx, row in df.iterrows():
    image_filename = row['image_name']
    label = row['label_name']
    x, y, width, height = row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']

    # Construct the full path to the image
    image_path = os.path.join(image_folder, image_filename)

    # Load image
    img = np.array(Image.open(image_path))
    img = cv2.resize(img, (128, 128))
    # Normalize image pixel values to range [0, 1]
    img = img / 255.0
    # Process image and label as needed
    # ... (you can resize, normalize, or preprocess the image)

    # Update the code to handle the bounding box and class label information
    # For simplicity, assuming a single class for each box
    label_info = [1.0, x, y, x + width, y + height, label]

    # Append the image and label to the lists
    images_list.append(img)
    if len(y_labels) == 0:
        y_labels = np.array([label_info])
    else:
        y_labels = np.append(y_labels, [label_info], axis=0)

# Convert lists to NumPy arrays
X_images = np.array(images_list)
# y_labels is already a NumPy array, no need to convert



x = x_input = layers.Input(shape=(128, 128, 3))

x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.BatchNormalization()(x) # size: 64x64

x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)  # size: 64x64

x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.BatchNormalization()(x)  # size: 32x32

x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.BatchNormalization()(x)  # size: 16x16

x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.BatchNormalization()(x) # size: 8x8x

# ---

x_prob = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', name='x_prob')(x)
x_boxes = layers.Conv2D(4, kernel_size=3, padding='same', name='x_boxes')(x)
x_cls = layers.Conv2D(10, kernel_size=3, padding='same', activation='sigmoid', name='x_cls')(x)

# ---

gate = tf.where(x_prob > 0.5, tf.ones_like(x_prob), tf.zeros_like(x_prob))
x_boxes = x_boxes * gate
x_cls = x_cls * gate

# ---

x = layers.Concatenate()([x_prob, x_boxes, x_cls])

model = tf.keras.models.Model(x_input, x)
model.summary()
# After training the model and generating predictions

idx_p = [0]
idx_bb = [1, 2, 3, 4]
idx_cls = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]






@tf.function
def loss_bb(y_true, y_pred):
    y_true = tf.gather(y_true, idx_bb, axis=-1)
    y_pred = tf.gather(y_pred, idx_bb, axis=-1)

    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return tf.reduce_mean(loss[loss > 0.0])

@tf.function
def loss_p(y_true, y_pred):
    y_true = tf.gather(y_true, idx_p, axis=-1)
    y_pred = tf.gather(y_pred, idx_p, axis=-1)

    loss = tf.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_sum(loss)

@tf.function
def loss_cls(y_true, y_pred):
    y_true = tf.gather(y_true, idx_cls, axis=-1)
    y_pred = tf.gather(y_pred, idx_cls, axis=-1)

    loss = tf.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_sum(loss)

@tf.function
def loss_func(y_true, y_pred):
    return loss_bb(y_true, y_pred) + loss_p(y_true, y_pred) + loss_cls(y_true, y_pred)

opt = tf.keras.optimizers.Adam(learning_rate=0.003)
model.compile(loss=loss_func, optimizer=opt)



def get_color_by_probability(p):
    if p < 0.3:
        return (1., 0., 0.)
    if p < 0.7:
        return (1., 1., 0.)
    return (0., 1., 0.)

def show_predict(X, y, threshold=0.1, ax=None):
    X = X.copy()
    grid_size = 16

    for mx in range(8):
        for my in range(8):
            channels = y[my][mx]
            prob, x1, y1, x2, y2 = channels[:5]

            # if prob < threshold we won't show anything
            if prob < threshold:
                continue

            color = get_color_by_probability(prob)
            # bounding box
            px, py = (mx * grid_size) + x1, (my * grid_size) + y1
            px2, py2 = (mx * grid_size) + x2, (my * grid_size) + y2

            # Draw bounding box
            cv2.rectangle(X, (int(px), int(py)), (int(px2), int(py2)), color, 1)

            # Display class label
            kls = np.argmax(channels[5:])
            cv2.putText(X, f'{kls}', (int(px + 2), int(py - 2)), cv2.FONT_HERSHEY_PLAIN, 0.7, (0.0, 0.0, 0.0))

    if ax is not None:
        ax.imshow(X)
    else:
        plt.imshow(X)
        plt.show()


# Assuming you have already loaded your model and defined the necessary functions
predictions = model.predict(X_images)

# Display all images and predictions
fig, axes = plt.subplots(1, len(X_images), figsize=(30, 10))
for i, (img, pred) in enumerate(zip(X_images, predictions)):
    ax = axes[i] if len(X_images) > 1 else axes  # Handle the case of a single image
    show_predict(img, pred, threshold=0.1, ax=ax)
    ax.set_title(f'Image {i+1}')
plt.show()


