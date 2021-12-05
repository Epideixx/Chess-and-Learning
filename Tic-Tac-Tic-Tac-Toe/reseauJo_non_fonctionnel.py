# -------------------------------------------------
#             Réseau neuronal MCTS
# -------------------------------------------------

# Ceci est une première version destinée à entraîner un réseau neuronal avec l'ensmeble des parties qui ont mené à la victoire

from time import time
import tensorflow as tf
import numpy as np
from math import floor
from random import shuffle

from tensorflow import keras

from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib
from MCTS_Jo import MCTS, convertdata

# Pour cette première version, les données seront sous la forme d'un plateau de jeu
# Le réseau prendra en entrée:
#       np.array([game.representation_reseau(symbol)])
# Et retournera en sortie :
#       - une politique de jeu vise à vis des coups possibles
#       - une estimation des chances de victoires

# ---        STEP 0 : Donner une représentation d'un plateau                ---

# Go faire 9 x 9 x 3 = une représentation pour X, une pour O, et une pour cases vides

# Rappel : la sortie sera une évaluation du plateau (easy) et une politique de jeu (moins easy)

# ---       STEP 1 : Création d'un réseau neuronal convolutif simple juste pour l'évaluation du plateau       ---

model_cnn1 = keras.models.Sequential()
model_cnn1.add(keras.layers.Conv2D(
    64, (3, 3), strides=(3, 3), activation='relu', input_shape=(9, 9, 3)))
model_cnn1.add(keras.layers.MaxPooling2D((3, 3)))

model_cnn1.add(keras.layers.Flatten(input_shape=(1, 1, 64)))
model_cnn1.add(keras.layers.Dense(16, activation='relu'))
model_cnn1.add(keras.layers.Dense(1, activation='sigmoid'))

# ---       STEP 2 : Entraînement du réseau neuronal avec tous les états qui ont mené à la victoire ou la défaite

# Go importer de la Data yessss
mcts = MCTS("O", 20)
start = time()
data = mcts.examples(15, convertdata_function=convertdata)
shuffle(data)
print(time() - start)

split = 0.8
(X_train, Y_train) = zip(*data[:floor(split*len(data))])

(X_test, Y_test) = zip(*data[floor(split*len(data)):])
X_train, X_test, Y_train, Y_test = np.array(list(
    X_train)), np.array(list(X_test)), np.array(list(Y_train)), np.array(list(Y_test))

# Puis on entraîne le modèle (ça marche pas...)
model_cnn1.summary()
model_cnn1.compile(optimizer='adam', metrics=[
                   'accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy())

history = model_cnn1.fit(X_train, Y_train, epochs=5,
                         validation_data=(X_test, Y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

if __name__ == '__main__':
    print('ok')
