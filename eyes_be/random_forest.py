import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Paths to the files and reading the labels
train_csv_path = r'C:\Users\calar\OneDrive\Bureau\DATACAMP\eyes_be\eyes-dataset\Training_Set\Training_Set\RFMiD_Training_Labels.csv'
train_labels = pd.read_csv(train_csv_path, dtype={'ID': str, 'Disease_Risk': str})
train_labels['ID'] = train_labels['ID'].apply(lambda x: f"{x}.png")

def load_and_preprocess_images_from_labels(labels, folder_path):
    images = []
    for img_name in tqdm(labels['ID'], desc="Loading and Preprocessing Images"):
        img_path = folder_path + '/' + img_name
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Normalization
            img_normalized = img / 255.0
            resized_img = cv2.resize(img_normalized, (64, 64))
            images.append(resized_img)
    return np.array(images)

train_folder_path = r'C:\Users\calar\OneDrive\Bureau\DATACAMP\eyes_be\eyes-dataset\Training_Set\Training_Set\Training'
train_images = load_and_preprocess_images_from_labels(train_labels, train_folder_path)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels['Disease_Risk'], test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train_encoded = pd.get_dummies(y_train).values
y_test_encoded = pd.get_dummies(y_test).values

# Flatten the images for SMOTE
X_train_flattened = X_train.reshape(X_train.shape[0], -1)

# Balancing classes with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_encoded_resampled = smote.fit_resample(X_train_flattened, y_train_encoded)

# Reshape images to their original shape after SMOTE
X_train = X_train_resampled.reshape(X_train_resampled.shape[0], 64, 64)

# Convert X_train to 3 channels
X_train = np.stack((X_train,)*3, axis=-1)
y_train_encoded = y_train_encoded_resampled

# Convert X_test to 3 channels
X_test = X_test.reshape(X_test.shape[0], 64, 64)
X_test = np.stack((X_test,)*3, axis=-1)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Fit the augmenter to our data
datagen.fit(X_train)

# Modifying the Network Architecture: Using VGG16 for fine-tuning
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(64, 64, 3)))

# Building the new model on top
headModel = baseModel.output
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(128, activation="relu")(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the layers of the base model
for layer in baseModel.layers:
    layer.trainable = False

# Change the loss function
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"])

# Train the model with data augmentation and a validation set
X_valid, X_test, y_valid_encoded, y_test_encoded = train_test_split(X_test, y_test_encoded, test_size=0.5, random_state=42)
history = model.fit(datagen.flow(X_train, y_train_encoded, batch_size=32),
                    validation_data=(X_valid, y_valid_encoded),
                    epochs=15,
                    steps_per_epoch=len(X_train) // 32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {accuracy}")

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Display the classification report
report = classification_report(np.argmax(y_test_encoded, axis=1), y_pred, zero_division=1)
print(report)
