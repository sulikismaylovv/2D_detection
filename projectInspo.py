#Import needed libraries
import numpy as np
import pandas as pd
import itertools
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import the Tensor Flow libraries
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import linear, relu, sigmoid
from sklearn.model_selection import ParameterGrid
import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#Import OS libraries
from pathlib import Path
import os.path
import os



def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

#################################################################From Assignments###########################################################################
def plot_loss_tf(history):
    fig,ax = plt.subplots(1,1, figsize = (4,3))
    widgvis(fig)
    ax.plot(history.history['loss'], label='training_loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()

def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
#################################################################End From Assignments###########################################################################


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           yticklabels=labels)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)



print("Current Working Directory:", os.getcwd())
dataset = 'data/images_dataset'

if os.path.exists(dataset):
    print("Path exists:", dataset)
else:
    print("Path does not exist:", dataset)

walk_through_dir(dataset)


image_dir = Path(dataset)
filepaths = list(image_dir.glob(r'**/*.jpeg'))+ list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

image_df = pd.concat([filepaths, labels], axis=1)



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
############################Code from Assignment################################
#View 20 random images
num_images = 20
m = image_df.shape[0]

fig, axes = plt.subplots(5, 4, figsize=(10, 10))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

widgvis(fig)
for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(m)
    ax.imshow(plt.imread(image_df.Filepath[random_index]))
    ax.set_title(f"Label: {image_df.Label[random_index]}")
    ax.set_axis_off()
    if i + 1 == num_images:
        break

fig.suptitle("Label, image", fontsize=14)
plt.show()
#########################End Code from Assignment###############################



train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

train_df = train_df[['Filepath', 'Label']]
test_df = test_df[['Filepath', 'Label']]



label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df['Label'])
val_labels_encoded = label_encoder.transform(train_df['Label'])

num_classes = len(label_encoder.classes_)
train_labels = to_categorical(train_labels_encoded, num_classes)
val_labels = to_categorical(val_labels_encoded, num_classes)

train_data, test_data, _, _ = train_test_split(
    train_df['Filepath'],
    train_labels,
    test_size=0.2,
    random_state=42
)
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    validation_split=0.25,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)


test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)



train_labels_one_hot = to_categorical(train_labels, num_classes)
val_labels_one_hot = to_categorical(val_labels, num_classes)


param_grid = {
    'batch_size': [32],
    'learning_rate': [0.001]
}


for params in ParameterGrid(param_grid):
    print(f"Training with hyperparameters: {params}")
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=params['batch_size'],
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=params['batch_size'],
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=params['batch_size'],
        shuffle=False
    )

    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(224, 224),
        layers.experimental.preprocessing.Rescaling(1. / 255),
    ])

    # Load the pretrained VGG16 model
    pretrained_model = VGG16(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False

    print(train_images[0][0].shape)

    ###################Inspired on Assignment Code##################################
    inputs = pretrained_model.input
    x = resize_and_rescale(inputs)
    x = Dense(256, activation='relu')(pretrained_model.output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    history = model.fit(
        train_images,
        steps_per_epoch=len(train_images),
        validation_data=val_images,
        validation_steps=len(val_images),
        epochs=30
    )
    results = model.evaluate(test_images, verbose=0)
    #####################End of Assignment Code#####################################

    print("    Test Loss: {:.5f}".format(results[0]))
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))

    plot_loss_tf(history)

    # Predict the label of the test_images
    pred = model.predict(test_images)
    prediction = pred
    pred = np.argmax(pred,axis=1)
    # Map the label
    labels = (train_images.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]

    # Display the result
    print(f'The first 5 predictions: {pred[:5]}')
    # Display largest prediction index
    print(f" Largest Prediction index: {np.argmax(pred)}")
    y_test = list(test_df.Label)

    ###############Print Classification Report######################################
    print(classification_report(y_test, pred))

    ###############Random images with Predicted vs True Labels######################
    random_index = np.random.randint(0, len(test_df) - 1, 15)
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 15),
                             subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(test_df.Filepath.iloc[random_index[i]]))
        if test_df.Label.iloc[random_index[i]] == pred[random_index[i]]:
            color = "green"
        else:
            color = "red"
        ax.set_title(f"True: {test_df.Label.iloc[random_index[i]]}\nPredicted: {pred[random_index[i]]}", color=color)
    plt.show()
    plt.tight_layout()



sample_image = train_images[0][0][0]
activation_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers])
activations = activation_model.predict(sample_image.reshape(1, 224, 224, 3))
layer_index = 5
layer_activation = activations[layer_index][0]
num_filters = layer_activation.shape[-1]
rows = int(np.sqrt(num_filters))
cols = int(np.ceil(num_filters / rows))

plt.figure(figsize=(16, 16))
for i in range(num_filters):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(layer_activation[:, :, i], cmap='viridis')
    plt.axis('off')

plt.show()