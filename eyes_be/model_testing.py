import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model('model.h5')

# Charger les données depuis Excel
test_labels = pd.read_csv(r'C:\Users\calar\OneDrive\Bureau\DATACAMP\eyes_be\eyes-dataset\Test_Set\Test_Set\RFMiD_Testing_Labels.csv', dtype={'ID': str, 'Disease_Risk': str})

# Ajouter l'extension .jpg (ou l'extension appropriée) à chaque valeur dans la colonne 'ID'
test_labels['ID'] = test_labels['ID'].apply(lambda x: f"{x}.png")

test_datagen = ImageDataGenerator(rescale=1./255)
test_images_path = 'C:/Users/calar/OneDrive/Bureau/DATACAMP/eyes_be/eyes-dataset/Test_Set/Test_Set/Test'

# Préparer les générateurs d'images
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_labels,
    directory='C:/Users/calar/OneDrive/Bureau/DATACAMP/eyes_be/eyes-dataset/Test_Set/Test_Set/Test',
    x_col="ID",
    y_col="Disease_Risk",
    class_mode="binary",
    target_size=(224,224),
    batch_size=1,
    shuffle=False  # Important pour la correspondance des prédictions avec les noms de fichiers
)

# Prédictions
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_classes = (predictions > 0.5).astype("int32")

# Évaluation
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

# Si vous voulez afficher des images avec leurs prédictions, vous pouvez utiliser le code que vous avez partagé dans votre message précédent.

import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Noms des fichiers d'image
filenames = test_generator.filenames

# Afficher quelques images avec leurs prédictions
num_images_to_show = 640 # Nombre d'images à afficher

# Choix aléatoire d'indices d'images
random_indices = random.sample(range(len(filenames)), num_images_to_show)

# Chemin vers les images
test_images_path = 'C:/Users/calar/OneDrive/Bureau/DATACAMP/eyes_be/eyes-dataset/Test_Set/Test_Set/Test'

'''for i in random_indices:
    img = mpimg.imread(test_images_path + '\\' + filenames[i])
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_classes[i]}, Actual: {test_generator.labels[i]}")
    plt.show()'''

nb=0
for i in random_indices:
    print(f"Predicted: {predicted_classes[i]}, Actual: {test_generator.labels[i]}")
    if predicted_classes[i] == test_generator.labels[i]:
        nb+=1
print("tu as eu ",nb," bonnes réponses sur ",num_images_to_show," images.")

