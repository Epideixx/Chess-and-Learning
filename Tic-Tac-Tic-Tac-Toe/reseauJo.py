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
from MCTS_Jo import MCTS, convertdata_dense, convertdata

# Pour cette première version, les données seront sous la forme d'un plateau de jeu avec le format de Benjamin
# Le réseau prendra en entrée:
#       np.array([game.representation_reseau(symbol)])
# Et retournera en sortie :
#       - une politique de jeu vise à vis des coups possibles
#       - une estimation des chances de victoires = Done

# ---        STEP 0 : Donner une représentation d'un plateau                ---

# On utilisera la représentation de Benjamin

# Rappel : la sortie sera une évaluation du plateau (easy) et une politique de jeu (moins easy)

# ---        STEP 1 : Création des données               ---

mcts = MCTS("O", 20)
start = time()
data = mcts.examples(150, convertdata_function=convertdata_dense)
shuffle(data)
print(time() - start)

split = 0.8
(X_train, Y_train, Z_train) = zip(*data[:floor(split*len(data))])

(X_test, Y_test, Z_test) = zip(*data[floor(split*len(data)):])
X_train, X_test, Y_train, Y_test, Z_train, Z_test = np.array(list(
    X_train)), np.array(list(X_test)), np.array(list(Y_train)), np.array(list(Y_test)), np.array(list(Z_train)), np.array(list(Z_test))

# ---       STEP 2 : Création d'un réseau neuronal dense pour l'évaluation seule du plateau       ---

model_dense_eval = keras.models.Sequential()
model_dense_eval.add(keras.Input(shape=(81*2+9*3+9,)))
model_dense_eval.add(keras.layers.Dense(512, activation='sigmoid'))
model_dense_eval.add(keras.layers.Dense(64, activation='sigmoid'))
model_dense_eval.add(keras.layers.Dense(1, activation='sigmoid'))

model_dense_eval.compile(
    optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

# ---       STEP 3 : Création d'un réseau neuronal dense pour l'évaluation des coups       ---

model_dense_coups = keras.models.Sequential()
model_dense_coups.add(keras.Input(shape=(81*2+9*3+9,)))
model_dense_coups.add(keras.layers.Dense(512, activation='sigmoid'))
model_dense_coups.add(keras.layers.Dense(128, activation='sigmoid'))
model_dense_coups.add(keras.layers.Dense(81, activation='sigmoid'))

model_dense_coups.compile(
    optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])


# ---       STEP 4 : Entraînement des réseaux neuronaux       ---


history = model_dense_eval.fit(X_train, Y_train, epochs=10, shuffle=True,
                               validation_data=(X_test, Y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

history_coup = model_dense_coups.fit(X_train, Z_train, epochs=10, shuffle=True,
                                     validation_data=(X_test, Z_test))

plt.plot(history_coup.history['accuracy'], label='accuracy')
plt.plot(history_coup.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
