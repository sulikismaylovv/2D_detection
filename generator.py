import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Create an ImageDataGenerator with desired augmentations
def create_image_data_generator():
    return ImageDataGenerator(
        rotation_range=20,       # Rotate the image within 20 degrees range
        width_shift_range=0.1,   # Shift the image width by a maximum of 10% 
        height_shift_range=0.1,  # Shift the image height by a maximum of 10%
        shear_range=0.1,         # Shear the image by 10%
        zoom_range=0.1,          # Zoom in/out within 10% range
        horizontal_flip=True,    # Allow horizontal flipping
        fill_mode='nearest',     # Fill in missing pixels after a shift or rotation
    )

# Function to save augmented images
def save_augmented_images(data_gen, directory_path, num_images=5):
    # Create the output directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    # Generate and save the augmented images
    for i, (image, _) in enumerate(data_gen):
        if i >= num_images:
            break
        for j in range(image.shape[0]):
            tf.keras.preprocessing.image.save_img(
                os.path.join(directory_path, f"augmented_image_{i}_{j}.jpg"),
                image[j]
            )

# Function to generate and save augmented images from a directory of images
def generate_augmented_images_from_directory(dataset_directory, output_directory, csv_file_path, num_augmented_images=3):
    # Read the CSV file into a DataFrame
    labels_df = pd.read_csv(csv_file_path)
    augmented_images_info = []

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create the ImageDataGenerator
    data_gen = create_image_data_generator()

    # Loop through all files in the CSV
    for index, row in labels_df.iterrows():
        # Get the file name, label, width, and height
        image_file = os.path.join('images_dataset', row['image_name'])  # Update path for original images
        label_name = row['label_name']
        image_width = row['image_width']
        image_height = row['image_height']

        # Full path to the image file
        file_path = os.path.join(dataset_directory, row['image_name'])

        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist.")
            continue

        # Load the image
        original_image = tf.keras.preprocessing.image.load_img(file_path)
        original_image = tf.keras.preprocessing.image.img_to_array(original_image)
        original_image = np.expand_dims(original_image, axis=0)  # Add batch dimension

        # Create an iterator for augmentation
        image_iterator = data_gen.flow(
            original_image,
            batch_size=1
        )

        # Generate and save the augmented images
        for i in range(num_augmented_images):
            # Generate augmented image
            augmented_image = next(image_iterator)[0].astype(np.uint8)
            augmented_image_file_name = f"{row['image_name'].split('.')[0]}_{label_name}_{i}.jpg"
            augmented_image_file_path = os.path.join('output', augmented_image_file_name)  # Update path for augmented images
            augmented_image_full_path = os.path.join(output_directory, augmented_image_file_name)

            # Save the augmented image
            tf.keras.preprocessing.image.save_img(
                augmented_image_full_path,
                augmented_image
            )
            print(f"Saved: {augmented_image_full_path}")

            # Append the info to the list, with updated image_name to include directory
            augmented_images_info.append({
                'label_name': label_name,
                'image_name': augmented_image_file_path,
                'image_width': image_width,
                'image_height': image_height
            })

    # Update the original DataFrame to include the folder prefix
    labels_df['image_name'] = labels_df['image_name'].apply(lambda x: os.path.join('images_dataset', x))

    # Create a DataFrame from the augmented images info
    augmented_images_df = pd.DataFrame(augmented_images_info)

    # Append this DataFrame to the updated original labels DataFrame
    all_images_df = pd.concat([labels_df, augmented_images_df], ignore_index=True)

    # Save the combined DataFrame to a new CSV file in the data directory
    augmented_labels_csv_path = os.path.join('data', 'augmented_labels.csv')
    all_images_df.to_csv(augmented_labels_csv_path, index=False)
    print(f"Augmented labels CSV file has been saved to {augmented_labels_csv_path}")
 

# Example usage:
dataset_directory = 'data/images_dataset'  # Replace with the path to your dataset
output_directory = 'data/output' 
csv_file_path = 'data/labels.csv'          # Replace with your actual CSV file path
# Replace with your desired output path
generate_augmented_images_from_directory(dataset_directory, output_directory, csv_file_path)


