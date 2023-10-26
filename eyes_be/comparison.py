# Importation des bibliothèques nécessaires
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Chemins des fichiers et lecture des labels
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

# Définition du modèle CNN
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

early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Entraînement du modèle CNN
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=1,  # Augmentez le nombre d'époques
    callbacks=[early_stop]
)


# Prédictions
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_classes = (predictions > 0.5).astype("int32")

# Évaluation
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

validation_loss, accuracy_CNN_validation = model.evaluate(validation_generator)

# Charger les images en fonction des labels
def load_images_from_labels(labels, folder_path):
    images = []
    missing_images = []  # Pour suivre les images qui ne peuvent pas être lues
    for img_name in tqdm(labels['ID'], desc="Loading Images"):
        img_path = folder_path + '/' + img_name
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(cv2.resize(img, (64, 64)))
        else:
            missing_images.append(img_name)
    if len(missing_images) > 0:
        print(f"Warning: Couldn't read {len(missing_images)} images. Missing images: {', '.join(missing_images)}")
    return np.array(images)

train_folder_path = r'C:\Users\calar\OneDrive\Bureau\DATACAMP\eyes_be\eyes-dataset\Training_Set\Training_Set\Training'
train_images = load_images_from_labels(train_labels, train_folder_path)
if train_images.size == 0:
    raise ValueError("No images were loaded. Check the file paths and integrity of the images.")


# Aplatir les images
train_images_flattened = train_images.reshape(train_images.shape[0], -1)

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(train_images_flattened, train_labels['Disease_Risk'], test_size=0.2, random_state=42)

# Entraîner le modèle de forêt aléatoire
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Prédictions
y_pred = rf.predict(X_test)

# Évaluer la performance
accuracy_RFC = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy_RFC}")
print(report)

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


from sklearn.svm import SVC

# Entraîner le modèle SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Prédictions
y_pred = svm.predict(X_test)

# Évaluer la performance
accuracy_SVM = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy_SVM}")
print(report)

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

# Entraîner le modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prédictions
y_pred = knn.predict(X_test)

# Évaluer la performance
accuracy_KNN = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy_KNN}")

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#gradient boosted trees
from sklearn.ensemble import GradientBoostingClassifier

# Entraîner le modèle GBT
gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
gbt.fit(X_train, y_train)

# Prédictions
y_pred = gbt.predict(X_test)

# Évaluer la performance
accuracy_GBT = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy_GBT}")
print(report)

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Chemins des dossiers de validation et de test
validation_folder_path = r'C:\Users\calar\OneDrive\Bureau\DATACAMP\eyes_be\eyes-dataset\Evaluation_Set\Evaluation_Set\Validation'
test_folder_path = r'C:\Users\calar\OneDrive\Bureau\DATACAMP\eyes_be\eyes-dataset\Test_Set\Test_Set\Test'

# Charger les images de validation et de test
validation_images = load_images_from_labels(validation_labels, validation_folder_path)
test_images = load_images_from_labels(test_labels, test_folder_path)

# Aplatir les images
validation_images_flattened = validation_images.reshape(validation_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)

# Convertir les labels en format numérique
y_validation = validation_labels['Disease_Risk'].astype(int)
y_test = test_labels['Disease_Risk'].astype(int)

# Évaluer la performance des modèles sur les ensembles de validation et de test
# Exemple pour le modèle Random Forest
y_pred_validation_rf = rf.predict(validation_images_flattened)
y_pred_test_rf = rf.predict(test_images_flattened)

y_pred_validation_rf = y_pred_validation_rf.astype(int)
y_pred_test_rf = y_pred_test_rf.astype(int)


accuracy_RFC_validation = accuracy_score(y_validation, y_pred_validation_rf)
accuracy_RFC_test = accuracy_score(y_test, y_pred_test_rf)

print(f"Random Forest Validation Accuracy: {accuracy_RFC_validation}")
print(f"Random Forest Test Accuracy: {accuracy_RFC_test}")

# Évaluation pour le modèle SVM
y_pred_validation_svm = svm.predict(validation_images_flattened)
y_pred_test_svm = svm.predict(test_images_flattened)

y_pred_validation_svm = y_pred_validation_svm.astype(int)
y_pred_test_svm = y_pred_test_svm.astype(int)


accuracy_SVM_validation = accuracy_score(y_validation, y_pred_validation_svm)
accuracy_SVM_test = accuracy_score(y_test, y_pred_test_svm)

print(f"SVM Validation Accuracy: {accuracy_SVM_validation}")
print(f"SVM Test Accuracy: {accuracy_SVM_test}")

# Évaluation pour le modèle k-NN
y_pred_validation_knn = knn.predict(validation_images_flattened)
y_pred_test_knn = knn.predict(test_images_flattened)

y_pred_validation_knn = y_pred_validation_knn.astype(int)
y_pred_test_knn = y_pred_test_knn.astype(int)


accuracy_KNN_validation = accuracy_score(y_validation, y_pred_validation_knn)
accuracy_KNN_test = accuracy_score(y_test, y_pred_test_knn)

print(f"k-NN Validation Accuracy: {accuracy_KNN_validation}")
print(f"k-NN Test Accuracy: {accuracy_KNN_test}")

# Évaluation pour le modèle GBT
y_pred_validation_gbt = gbt.predict(validation_images_flattened)
y_pred_test_gbt = gbt.predict(test_images_flattened)

y_pred_validation_gbt = y_pred_validation_gbt.astype(int)
y_pred_test_gbt = y_pred_test_gbt.astype(int)


accuracy_GBT_validation = accuracy_score(y_validation, y_pred_validation_gbt)
accuracy_GBT_test = accuracy_score(y_test, y_pred_test_gbt)

print(f"GBT Validation Accuracy: {accuracy_GBT_validation}")
print(f"GBT Test Accuracy: {accuracy_GBT_test}")


# Évaluation du modèle CNN
loss, accuracy_CNN = model.evaluate(test_generator)

# Ajout de l'accuracy du CNN à la liste des accuracies
model_names = ['CNN']  # Initialise avec CNN, ajoutez d'autres modèles plus tard
accuracies = [accuracy_CNN]  # Initialise avec accuracy_CNN, ajoutez d'autres accuracies plus tard

# Insérez les nouvelles précisions dans vos listes pour la comparaison finale des modèles
model_names.extend(['Random Forest Validation', 'Random Forest Test',
               'SVM Validation', 'SVM Test',
               'k-NN Validation', 'k-NN Test',
               'GBT Validation', 'GBT Test'])
accuracies.extend([accuracy_RFC_validation, accuracy_RFC_test,
                   accuracy_SVM_validation, accuracy_SVM_test,
                   accuracy_KNN_validation, accuracy_KNN_test,
                   accuracy_GBT_validation, accuracy_GBT_test])

# Continuez avec la comparaison finale des modèles
plt.figure(figsize=(10, 8))
plt.bar(model_names, accuracies, color='green', width=0.4)
plt.ylabel('Accuracy')
plt.title('Comparaison des modèles')
plt.xticks(rotation=45)  # Option pour rendre les étiquettes lisibles
plt.show()