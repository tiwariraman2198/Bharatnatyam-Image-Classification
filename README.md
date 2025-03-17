# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


#loading the dataset
from google.colab import drive
drive.mount('/content/drive')
dataset_path = '/content/drive/MyDrive/MyDataset/Bharatnatyam_Dataset'


# Listing the contents of Google Drive
import os

drive_base_path = '/content/drive/MyDrive/MyDataset/Bharatnatyam_Dataset'
print("Files and folders in My Drive:")
print(os.listdir(drive_base_path))


# Getting the list of classes (folders)
class_names = os.listdir(dataset_path)
print("Classes found:", class_names)

# Creating a dictionary to map class names to labels
class_labels = {class_name: index for index, class_name in enumerate(class_names)}
print("Class labels:", class_labels)


# Define image size
image_size = (128, 128)  # You can adjust this based on your model's requirements

X = []
y = []

# Load images and labels
for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)  # Load the image
        img = cv2.resize(img, image_size)  # Resize the image
        img = img / 255.0  # Normalize pixel values
        X.append(img)
        y.append(class_labels[class_name])  # Get the label for the class

X = np.array(X)
y = np.array(y)

print("Loaded images shape:", X.shape)
print("Labels shape:", y.shape)


import matplotlib.pyplot as plt
import random

# Setting the number of images to display per class
num_images_to_display = 5

# Creating a figure to display images
plt.figure(figsize=(15, len(class_names) * 2))  # Adjust height based on number of classes

# Iterating through each class
for class_index, class_name in enumerate(class_names):

    # Finding all image indices for the current class
    class_indices = np.where(y == class_index)[0]

    # Randomly select a few image indices
    selected_indices = random.sample(list(class_indices), min(num_images_to_display, len(class_indices)))

    # Display the class label
    plt.subplot(len(class_names), num_images_to_display + 1, class_index * (num_images_to_display + 1) + 1)  # Position for the class label
    plt.title(class_name, fontsize=16)  # Display the class label
    plt.axis('off')  # Turn off axis labels

    for i, img_index in enumerate(selected_indices):
        plt.subplot(len(class_names), num_images_to_display + 1, class_index * (num_images_to_display + 1) + (i + 2))  # Position for images
        plt.imshow(X[img_index])
        plt.axis('off')  # Turn off axis labels

plt.tight_layout()
plt.show()

#splitting the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)


#model building

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


#training the model
from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Using data augmentation
datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20)

# Fit the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=25)  # Adjust epochs as needed

#evaluation of model
# Optional: Plot loss history
plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


#classification report
from sklearn.metrics import classification_report

# Predicting the labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

# Generating the classification report
report = classification_report(y_test, y_pred_classes, target_names=class_names)

# Printing the classification report
print(report)
