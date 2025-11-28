



from genderModelAge import create_age_model
from dataPrepationAge import dataPreparation, split_data

#Keras qui sauvegarde le modèle après chaque epoch si c’est le meilleur
from tensorflow.keras.callbacks import ModelCheckpoint

#data augmentation (rotation, translation, zoom…) à la volée.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#les courbes de loss et d’accuracy après l’entraînement.
import matplotlib.pyplot as plt

DATASET_PATH = ["UTKFace/", "combined_faces/"]
MODEL_PATH = "age_model.h5"
input_shape = (200, 200, 1)

#sauvegarde le modèle uniquement si la loss diminue (save_best_only=True).
#on surveille la loss du training pour décider de sauvegarder.
#liste des callbacks pour l’entraînement (ici, juste checkpointer).
checkpointer = ModelCheckpoint(
    MODEL_PATH,
    #métrique le callback doit surveiller. Ici, loss signifie que le callback surveille la perte d’entraînement.
    monitor='val_loss',
    #messag de sauvgarde
    verbose=1,
    save_best_only=True,
    #True → ne sauvegarde que les poids. False → sauvegarde le modèle entier.
    save_weights_only=False,
    #Keras choisit automatiquement selon la métrique surveillée.
    mode='auto'
)
callback_list = [checkpointer]

#create the model
model = create_age_model(input_shape)

model.compile(
    #Algorithme d’optimisation qui ajuste les poids du modèle pour minimiser la perte.
    #Adam est très populaire car il est efficace et rapide pour la plupart des problèmes.
    optimizer='adam',
    #La fonction de perte que le modèle va essayer de minimiser.
    #categorical_crossentropy est utilisée pour la classification multi-classes où les étiquettes sont encodées en one-hot.
    loss='categorical_crossentropy',
    #Keras de calculer la précision (accuracy) à chaque époque pour suivre la performance du modèle.
    metrics=['accuracy']
)

pixels, age_labels  = dataPreparation(DATASET_PATH, aug=0)
x_train, x_test, y_train, y_test = split_data(
    pixels, age_labels , test_size=0.2
    #, random_state=100 meme valeur repter
)

#préparer les images pour l’entraînement et peut appliquer des augmentations en temps réel
train_datagen = ImageDataGenerator(
    #rotation aléatoire des images de ±10°
    rotation_range=10,
    #déplacement horizontal aléatoire jusqu’à 10% de la largeur
    width_shift_range=0.1,
    #déplacement vertical aléatoire jusqu’à 10% de la hauteur
    height_shift_range=0.1,
    #zoom aléatoire jusqu’à ±10%
    zoom_range=0.1,
    
    #,horizontal_flip=True
    horizontal_flip=True
)

val_datagen = ImageDataGenerator()

#Création du générateur d’entraînement
#flow() transforme tes tableaux x_train et y_train en générateur de batchs.
train_gen = train_datagen.flow(
    x_train, y_train,
    #Définition : nombre d’images que le modèle traite avant de mettre à jour ses poids.
    #batch_size=64 → nombre d’images traitées à chaque étape d’entraînement
    batch_size=64,
    #shuffle=True → mélange aléatoire des images à chaque époque
    shuffle=True
)

val_gen = val_datagen.flow(
    x_test, y_test,
    ##batch_size=64 → 64 images × 200×200×1 × 4 bytes ≈ 10 Mo
    #Ton GPU charge seulement ces 10 Mo pour calculer le gradient et mettre à jour les poids.
    batch_size=64,#32,16  cest valeur a tester puis que on peut pas dire quelle est la meilleur
    shuffle=False
)


history = model.fit(
    #générateur d’images pour l’entraînement (avec augmentation et batchs)
    train_gen,
    #validation_data=val_gen : générateur pour la validation (images réelles, pas d’augmentation).
    validation_data=val_gen,
    #epochs=15 : nombre de fois que le modèle verra toutes les images d’entraînement.
    #batch_size=64 → 64 images × 200×200×1 × 4 bytes ≈ 10 Mo
    #Ton GPU charge seulement ces 10 Mo pour calculer le gradient et mettre à jour les poids.
    epochs=15,
    #Dans ton cas, callback_list contient ModelCheckpoint pour sauvegarder automatiquement le meilleur modèle.
    callbacks=callback_list
)


#'loss' → perte d’entraînement par epoch
train_loss = history.history['loss']
#'val_loss' → perte sur les données de validation 
val_loss = history.history['val_loss']
#'accuracy' → précision d’entraînement 
train_acc = history.history['accuracy']
#'val_accuracy' → précision sur les données de validation
val_acc = history.history['val_accuracy']

#figsize=(15,7) → définit la taille de la figure.
#plt.subplots(ncols=2) → crée 2 graphiques côte à côte (colonnes).
fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
#ax.ravel() → transforme la liste d’axes en tableau plat pour accéder facilement à ax[0] et ax[1].
ax = ax.ravel()



# LOSS
ax[0].plot(train_loss, label="Train Loss", marker='o')
#marker='o' → ajoute un point pour chaque epoch.
ax[0].plot(val_loss, label="Val Loss", marker='o')
#xlabel / ylabel → nomme les axes.
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Categorical Crossentropy")
#legend() → ajoute la légende pour distinguer train/validation.
ax[0].legend()

# ACCURACY
ax[1].plot(train_acc, label="Train Accuracy", marker='o')
ax[1].plot(val_acc, label="Val Accuracy", marker='o')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

#suptitle → titre global pour toute la figure.
plt.suptitle("CNN age_labels  Model - Loss & Accuracy per Epoch", fontsize=16)
#plt.show() → affiche le graphique.
plt.show()
