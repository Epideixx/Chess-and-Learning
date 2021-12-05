import argparse  # ?
import os
import shutil  # File operations
import random
import numpy as np
import math
import sys
import time
import abc  # Abstract Base Class
import joblib  # Save data structures in files

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Reshape, Flatten, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import activations, regularizers
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import wandb
from wandb.keras import WandbCallback


class CustomModel():

    def getInputTargets(self, examples):
        """
        Preprocessing data for compatibility with tensorflow

        INPUT:
            examples (list of tuples) : [board, policy, evaluation]

        OUTPUT :
            input_board, target_pis, target_vs => Processed data
        """

        ### TO COMMENT ###
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        pi = []
        for move in target_pis:
            if type(list(move.keys())[0]) == int:
                temp = np.zeros(7)
                for (u, v) in move.items():
                    temp[u] = v
            else:
                temp = np.zeros((9, 9))
                for (u, v) in move.items():
                    temp[int(u[0]), int(u[1])] = v
            pi.append(temp)
        target_pis = np.asarray(pi)
        target_vs = np.asarray(target_vs)

        return input_boards, target_pis, target_vs

    def predictBatch(self, boards):
        """
        Estimates policies and evaluations

        INPUT: 
            boards (dict) : dict of boards

        OUTPUTs:
            finalpi (dict) : key = board, value = movespi(dict of move : proba)
            finalv (dict) : key = board, value = evaluation of the board (float)
        """

        numpyBoards = np.asarray([boards[b].representation for b in boards])

        pi, v = self.prediction(numpyBoards)  # Prediction for every board

        finalpi = dict()
        finalv = dict()

        for e, board in enumerate(boards):  # (e = position in dict, board = key)
            movespi = dict()  # for each board, there is a policy

            # We look at each possible move from this board
            for m in boards[board].get_legal_moves():
                if type(m) == int:
                    movespi[m] = pi[e, m]
                else:  # WHY ?
                    movespi[m] = pi[e, m[0], m[1]]

            total = sum(movespi.values())
            if total <= 1E-5:  # No good move, avoid errors, TO WATCH
                print("Shit")
                for move in movespi:
                    movespi[move] = 1
            finalpi[board] = movespi
            finalv[board] = v[e][0]  # evaluation of the e-th board
        return finalpi, finalv

    @abc.abstractmethod
    def prediction(self, numpyBoards):
        """
        Estimation de la politique et de la valeur avec le modèle
        """
        return


class NeuralNetwork(CustomModel):  # WHY CustomModel here ?

    def residual_block(self, x, filters, kernel_size=3):
        """
        Creation of the Residual block which will be usefull for the ResNet
        """
        y = Conv2D(kernel_size=kernel_size, filters=filters, padding="same")(x)
        y = BatchNormalization()(y)
        y = Activation(activations.relu)(y)
        y = Conv2D(kernel_size=kernel_size, filters=filters, padding="same")(y)
        y = BatchNormalization()(y)
        y = Add()([x, y])
        y = Activation(activations.relu)(y)
        return y

    def __init__(self):
        """ 
        Initialization of the Neural Network
        """

        self.config = {
            'learning_rate': 0.001,
            'epochs': 1,
            'batch_size': 256,
            'filters': 256,
            'residualDepth': 10
        }

        self.inputs = Input(shape=(7, 6, 2))
        # Shape of the game :
        #  - 7 columns
        #  - 6 lines
        #  - 2 players

        # First Layer

        x = Conv2D(filters=self.config['filters'], kernel_size=(
            3, 3), padding="same")(self.inputs)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # Residual layers

        for _ in range(self.config['residualDepth']):
            x = self.residual_block(x, self.config['filters'], kernel_size=3)

        # Then we split the NN into two parts : EVALUATION and POLICY

        # Evaluation part
        valueHead = Conv2D(filters=8, kernel_size=(1, 1), padding="same")(x)
        valueHead = BatchNormalization()(valueHead)
        valueHead = Activation(activations.relu)(valueHead)
        valueHead = Flatten()(valueHead)
        valueHeadLastLayer = Dense(256, activation="relu")(valueHead)
        valueHeadLastLayer = Dropout(0.3)(valueHeadLastLayer)
        valueHead = Dense(1, activation="tanh", name="v")(valueHeadLastLayer)

        # Policy part
        policyHead = Conv2D(filters=32, kernel_size=(1, 1), padding="same")(x)
        policyHead = BatchNormalization()(policyHead)
        policyHead = Activation(activations.relu)(policyHead)
        policyHeadLastLayer = Flatten()(policyHead)
        policyHead = Dropout(0.3)(policyHeadLastLayer)
        policyHead = Dense(7, activation="softmax", name="pi")(
            policyHead)  # 7 possible moves

        self.model = Model(inputs=self.inputs, outputs=[
                           policyHead, valueHead, policyHeadLastLayer, valueHeadLastLayer])  # policyHeadLastLayer and valueHeadLastLayer for XGBoost
        self.compile()

        config = wandb.config  # Save config

    def compile(self, loss_weights=[1, 1, 1, 1]):
        """
        compile the model
        """
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error', None, None], optimizer=Adam(
            self.config['learning_rate']), loss_weights=loss_weights)

    def start_wandb(self, resume):
        """
        Save parameters of the model with wandb
        """
        wandb.init(project='alphazero', entity='naaranja',
                   config=self.config, resume=resume)

    def train(self, examples, warm_start=False):
        """
        Train the model
        """
        input_boards, target_pis, target_vs = self.getInputTargets(
            examples)  # Preprocessing data
        if warm_start:
            self.model.fit(x=input_boards, y=[target_pis, target_vs, np.zeros(target_vs.shape), np.zeros(
                target_vs.shape)], batch_size=self.config['batch_size'], epochs=1)
        else:
            self.model.fit(x=input_boards, y=[target_pis, target_vs, np.zeros(target_vs.shape), np.zeros(
                target_vs.shape)], batch_size=self.config['batch_size'], epochs=self.config['epochs'], callbacks=[WandbCallback()])
        # MAKE SOME RESEARCH ON CALLBACKS AND WANDBCALLBACK

    def prediction(self, numpyBoards):
        """
        Makes a prediction

        INPUT:
            numpyBoards : board as an nparray representing a state

        OUTPUTS:
            policy (dict)
            value (float)
        """
        return self.model.predict(numpyBoards)[:2]

    def save_checkpoint(self, folder, filename):
        """
        Save model
        """
        filepath = os.path.join(folder, filename)
        self.model.save(filepath)

    def load_checkpoint(self, folder, filename):
        """
        Load model
        """
        filepath = os.path.join(folder, filename)
        self.model = tf.keras.models.load_model(filepath)


class NeuralXGBoost(CustomModel):  # WHY CustomModel here ?

    """
    NeuralXGBoost uses intermediate outputs of the neural network as an input
    Using alternative outputs thanks to the XGBoost is optional by using parameters use_pi and use_v.
    """

    def __init__(self, use_pi=True, use_v=True):
        self.use_pi = use_pi
        self.use_v = use_v
        self.NN = NeuralNetwork()
        self.NN.load_checkpoint(
            folder='data', filename='daily_donkey/save48.h5')  # Filename may change
        if self.use_pi:  # First XGBoost for policy
            self.xgb_pi = xgb.XGBClassifier(
                use_label_encoder=False, colsample_bytree=0.5, n_estimators=100)
        if self.use_v:  # Second XGBoost for evaluation
            self.xgb_v = xgb.XGBRegressor(
                colsample_bytree=0.5, n_estimators=100)

    def train(self, examples):
        """
        Train XGBoost
        """

        # TO COMMENT A BIT MORE
        input_boards, target_pis, target_vs = self.getInputTargets(examples)
        policyHeadLastLayer, valueHeadLastLayer = [], []
        for i in range(1+len(input_boards)//100000):
            print(f'Step #{i}')
            data = input_boards[i *
                                100000:min((i+1)*100000, len(input_boards)+1)]
            p, v = self.NN.model.predict(data)[2:]  # Intermediate outputs
            policyHeadLastLayer += list(p)
            valueHeadLastLayer += list(v)
        policyHeadLastLayer = np.array(policyHeadLastLayer)
        valueHeadLastLayer = np.array(valueHeadLastLayer)

        target_pi_class = []
        for i in range(len(target_pis)):
            target_pi_class.append(np.argmax(target_pis[i]))

        if self.use_pi:
            # Train XGBoost policy
            self.xgb_pi.fit(policyHeadLastLayer,
                            target_pi_class, eval_metric='logloss')
            joblib.dump(self.xgb_pi, 'data/xgb_pi')  # save it

        if self.use_v:
            # Train XGBoost evaluation
            self.xgb_v.fit(valueHeadLastLayer, target_vs)
            joblib.dump(self.xgb_v, 'data/xgb_v')  # save it

    def prediction(self, numpyBoards):
        """
        Makes a prediction
        """
        pi, v, policyHeadLastLayer, valueHeadLastLayer = self.NN.model.predict(
            numpyBoards)
        if self.use_pi:
            pi = self.xgb_pi.predict_proba(policyHeadLastLayer)
        if self.use_v:
            v = self.xgb_v.predict(valueHeadLastLayer)
            v = [[e] for e in v]
        return pi, v

    def load(self, xgb_folder, nn_folder, nn_filename):
        """
        Load the XGBoost model
        """
        self.NN.load_checkpoint(folder=nn_folder, filename=nn_filename)
        self.xgb_pi = joblib.load(xgb_folder+'/xgb_pi')
        self.xgb_v = joblib.load(xgb_folder+'/xgb_v')
