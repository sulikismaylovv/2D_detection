import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import pandas as pd
import random
from data_preprocessing import load_annotations, split_data
from data_preparation import create_generators

# Assuming the model and history are saved during the training process
def load_best_model():
    # Placeholder for your model loading code
    # For example, if you saved your model as 'best_model.h5':
    model = tf.keras.models.load_model('models/model_1710169839.1884742.h5')
    return model

def evaluate_model(model, test_images):
    results = model.evaluate(test_images)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def predict_random_image(model, image_dir, csv_path, input_shape=(128, 128)):
    bbox_annotations = pd.read_csv(csv_path)
    image_files = bbox_annotations['image_name'].unique().tolist()
    random_image = random.choice(image_files)

    img = load_img(os.path.join(image_dir, random_image), target_size=input_shape)
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array_expanded)
    
    # Detailed explanation of prediction
    print("Raw prediction output (probabilities for each class):", prediction)
    
    # For each class, print the probability
    label_encoder = LabelEncoder()
    bbox_annotations['label_encoded'] = label_encoder.fit_transform(bbox_annotations['label_name'])
    encoded_labels = label_encoder.classes_
    print("Predicted probabilities by class:")
    for i, prob in enumerate(prediction[0]):
        print(f"{encoded_labels[i]}: {prob*100:.2f}%")

    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class_index)[0]

    actual_class_name = bbox_annotations[bbox_annotations['image_name'] == random_image]['label_name'].iloc[0]

    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title(f"Predicted Class: {predicted_label}, Actual Class: {actual_class_name}")
    plt.show()


# Replace with your actual test dataset

# Assuming these paths are correct, adjust if necessary
image_dir = 'data/images_dataset'
csv_path = 'data/labels.csv'

# Load and preprocess data
bbox_annotations = load_annotations(csv_path)
train_df, test_df = split_data(bbox_annotations)

# Initialize test_images data generator
_, test_images = create_generators(train_df, test_df, image_dir)

best_model = load_best_model()
if best_model:
    evaluate_model(best_model, test_images)

    # Assuming 'history' is saved and loaded similarly to the model
    # history = load_history()  # Placeholder for your history loading code
    # plot_history(history)

    # Predicting on random images
    for i in range(0):
        predict_random_image(best_model, image_dir, csv_path)
else:
    print("Model not found or could not be loaded.")
