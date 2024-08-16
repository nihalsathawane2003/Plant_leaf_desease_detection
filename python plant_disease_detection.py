import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def load_and_preprocess_dataset(dataset_path):
    X_all = []
    y_all = []

    class_directories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for class_dir in class_directories:
        class_path = os.path.join(dataset_path, class_dir)

        for filename in os.listdir(class_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(class_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Adjust for grayscale if needed
                image = cv2.resize(image, (256, 256))
                X_all.append(image)
                y_all.append(class_dir)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Normalize pixel values to [0, 1]
    X_all = X_all / 255.0

    label_encoder = LabelEncoder()
    y_all_encoded = label_encoder.fit_transform(y_all)
    class_names = label_encoder.classes_  # Get the class names

    return X_all, y_all_encoded, class_names

# Set the path to the PlantVillage dataset directory
dataset_path = 'C:\\Users\\Nihal\\Desktop\\Plant leaf detection\\Backend\\Test'

# Load and preprocess the dataset
X_all, y_all_encoded, class_names = load_and_preprocess_dataset(dataset_path)

# Print all unique class names
print("Class Names:", class_names)

# Shuffle the data
X_all, y_all_encoded = shuffle(X_all, y_all_encoded, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all_encoded, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Design the CNN model
num_classes = len(class_names)
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Add dropout for regularization
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model and store the history
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=20,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Save the entire model in the native Keras format
model.save('C:\\Users\\Nihal\\Desktop\\Plant leaf detection\\Backend\\plant_disease_model.keras')
print("Model saved successfully!")

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()
