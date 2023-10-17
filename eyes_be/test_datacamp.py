import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Chemins des fichiers
train_csv_path = r'C:\Users\calar\OneDrive\Bureau\DATACAMP\eyes_be\eyes-dataset\Training_Set\Training_Set\RFMiD_Training_Labels.csv'
validation_csv_path = r'C:\Users\calar\OneDrive\Bureau\DATACAMP\eyes_be\eyes-dataset\Evaluation_Set\Evaluation_Set\RFMiD_Validation_Labels.csv'
test_csv_path = r'C:\Users\calar\OneDrive\Bureau\DATACAMP\eyes_be\eyes-dataset\Test_Set\Test_Set\RFMiD_Testing_Labels.csv'

train_labels = pd.read_csv(train_csv_path, dtype={'ID': str, 'Disease_Risk': str})
validation_labels = pd.read_csv(validation_csv_path, dtype={'ID': str, 'Disease_Risk': str})
test_labels = pd.read_csv(test_csv_path, dtype={'ID': str, 'Disease_Risk': str})

train_labels['ID'] = train_labels['ID'].apply(lambda x: f"{x}.png")
validation_labels['ID'] = validation_labels['ID'].apply(lambda x: f"{x}.png")
test_labels['ID'] = test_labels['ID'].apply(lambda x: f"{x}.png")

# Créer un générateur d'images avec augmentation des données
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(dataframe=train_labels, directory='C:/Users/calar/OneDrive/Bureau/DATACAMP/eyes_be/eyes-dataset/Training_Set/Training_Set/Training', x_col="ID", y_col="Disease_Risk", class_mode="binary", target_size=(224,224), batch_size=32)
validation_generator = validation_datagen.flow_from_dataframe(dataframe=validation_labels, directory='C:/Users/calar/OneDrive/Bureau/DATACAMP/eyes_be/eyes-dataset/Evaluation_Set/Evaluation_Set/Validation', x_col="ID", y_col="Disease_Risk", class_mode="binary", target_size=(224,224), batch_size=32)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_labels, directory='C:/Users/calar/OneDrive/Bureau/DATACAMP/eyes_be/eyes-dataset/Test_Set/Test_Set/Test', x_col="ID", y_col="Disease_Risk", class_mode="binary", target_size=(224,224), batch_size=32)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])


opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,  # Augmentez le nombre d'époques
    callbacks=[early_stop]
)


test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

print(train_labels['Disease_Risk'].value_counts())
print(validation_labels['Disease_Risk'].value_counts())
print(test_labels['Disease_Risk'].value_counts())

model.save('model.h5')