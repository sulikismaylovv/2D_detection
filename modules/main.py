
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder


# Assuming other modules are refactored as well
from data_preparation import create_image_data_generators, plot_loss_tf, generate_augmented_images
from data_preprocessing import load_annotations, split_data

def compute_weights(y):
    """
    Computes class weights for balanced training.
    
    Args:
        y (pd.Series or np.array): Target class labels for the training data.
        
    Returns:
        A dictionary mapping class indices to their weight.
    """
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)
    classes = np.unique(encoded_y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=encoded_y)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    return class_weights_dict

def create_model(num_classes, input_shape=(256, 256, 3), fine_tune=5):
    """
    Creates a custom model for 2D detection.

    Args:
        input_shape (tuple): The shape of the input images. Defaults to (256, 256, 3).
        num_classes (int): The number of classes for classification. Defaults to 3.
        fine_tune (int): The number of layers to fine-tune in the base model. Defaults to 5.

    Returns:
        tf.keras.Model: The created model.

    """
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

def train_and_evaluate_model(train_data, test_data, num_classes=3):
    """Trains and evaluates the model using the provided data.
    
    Args:
        train_data: The training data generator.
        test_data: The testing data generator.
        num_classes (int): The number of classes in the dataset.

    Returns:
        The trained model and its training history.
    """
    num_classes_x = 3
    model = create_model(num_classes_x)
    model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_data,
        epochs=30,
        validation_data=test_data,
        callbacks=[early_stopping]
    )
    return model, history



def run_kfold_and_select_best_model(k=5):
    kfold = KFold(n_splits=k, shuffle=True)
    best_accuracy = 0
    best_model = None
    

    for fold, (train_idx, test_idx) in enumerate(kfold.split(annotations), start=1):
        print(f"Training on fold {fold}...")
        train_df, test_df = annotations.iloc[train_idx], annotations.iloc[test_idx]
        train_images, test_images = create_image_data_generators(preprocess_func, image_dir, train_df, test_df)
                    
        # Compute class weights for balanced training
        class_weights_dict = compute_weights(train_df['label_name'])
        print(f"Class weights: {class_weights_dict}")
                            
        model, history = train_and_evaluate_model(train_images, test_images, class_weights_dict)
                
        max_val_accuracy = max(history.history['val_accuracy'])
        if max_val_accuracy > best_accuracy:
            best_accuracy = max_val_accuracy
            best_model = model
            best_history = history
    
    return best_model, best_history

if __name__ == '__main__':
    num_classes = 3
    image_dir = 'data/'  # Updated to more explicitly define image directory
    csv_path = 'data/augmented_labels.csv'  # Updated for clarity
    
    annotations = load_annotations(csv_path)
    # Uncomment this if code is running on LINUX or MacOS
    annotations['image_name'] = annotations['image_name'].str.replace('\\', '/', regex=False)
    train_df, test_df = split_data(annotations)
    
    preprocess_func = tf.keras.applications.vgg16.preprocess_input
    #train_images, test_images = create_image_data_generators(preprocess_func, image_dir, train_df, test_df)
    # Generate and save augmented images
    #generate_augmented_images(train_images, 'data/output', 5)
    
    
    model, history = run_kfold_and_select_best_model(5)

    # Plot training and validation loss
    plot_loss_tf(history)
    
    # Save the trained model
    current_time = time.strftime("%d%m-%H-%M", time.localtime())
    model.save(f'models/model_{current_time}.h5')
    
    print("Model training and evaluation completed.")
