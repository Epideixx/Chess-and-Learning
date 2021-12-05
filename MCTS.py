# Enable to get some information when getting some error, or debugging...
import logging
import math
import time
import copy

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)  # Create the logger object

# s means "state"
# a means "action"


class MCTS():
    def __init__(self, cpuct):
        self.cpuct = cpuct  # exploration coefficient
        self.Qsa = {}  # Mean value of the next state ?
        self.Nsa = {}  # Number of time we took action a at state s
        self.Ns = {}  # Number of time we passed through state s
        self.Ps = {}  # Policy
        self.Es = {}  # Value of final states

        self.boardToPredict = None
        self.SAhistory = []  # History of every move

    def copy(self):
        """
        Make a deep copy of the MCTS
        """
        return copy.deepcopy(self)

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Returns the probabilities of the possible moves depending of the temperature
        For a temperature = 0, return the best move with a probability of 1

        INPUTS :
            canonicalBoard () : current Board
            temp (float) : float between 0 and 1 expressing the temperature

        OUTPUTS :
            probs (dict) : key = move/action, value = probability

        """
        s = canonicalBoard.fen()  # fen => string representation
        counts = {a: self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in self.Ps[s]}  # action : nb of time action a was choosen from s

        if temp == 0:
            bestN = max(counts.values())  # Max number of passage
            bestAs = {key: value for (key, value)
                      in counts.items() if value == bestN}  # Actions related to this bestN
            d = {i: e for i, e in enumerate(list(bestAs.keys()))}

            # Choose one of those actions
            bestA = np.random.choice(list(d.keys()))
            bestA = d[bestA]
            probs = {bestA: 1}  # Best action with a proba of 1
            return probs

        for move in counts:
            counts[move] = counts[move]**(1./temp)
        counts_sum = float(sum(counts.values()))
        # Actions with their probability
        probs = {move: x/counts_sum for (move, x) in counts.items()}
        return probs

    def selection(self, canonicalBoard):
        """
        Selection of a leaf by descending the MCTS 

        INPUT :
            canonicalBoard () : current state

        OUTPUT :
            Board which corresponds to a leaf state or None if final state is reached
        """
        self.SAhistory = []
        return self.findNext(canonicalBoard)

    def findNext(self, canonicalBoard):
        """
        Recursive descent to find a leaf node

        INPUT :
            canonicalBoard () : current state

        OUTPUT :
            Board which corresponds to a leaf state if to predict, else None
        """

        s = canonicalBoard.fen()  # fen => string representation

        # If we don't know the result of state s, we add the result
        if s not in self.Es:
            self.Es[s] = canonicalBoard.result()

        # If the node is final
        if self.Es[s] != 0:
            self.boardToPredict = canonicalBoard
            self.backpropagation(None, self.Es[s])  # (pi, v)
            return None

        # Else, if state s has not been predicted/visited yet
        if s not in self.Ps:
            self.boardToPredict = canonicalBoard
            self.Ns[s] = 0
            return self.boardToPredict

        cur_best = -float('inf')
        best_move = -1

        # Else, we continue our descent by choosing the one which maximizes uct
        for a in self.Ps[s]:

            # If action a has already been choosen once
            if (s, a) in self.Qsa:  # (s, a) => action a from state s !
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * \
                    math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])  # uct score

            # Else, if a is choosen for the first time
            else:
                u = self.cpuct * self.Ps[s][a] * \
                    math.sqrt(self.Ns[s] + EPS)  # uct score

            if u > cur_best:
                cur_best = u
                best_move = a

        a = best_move

        # we add the action to the history of every move
        self.SAhistory.append((s, a))

        canonicalBoard = canonicalBoard.copy()
        canonicalBoard.push(best_move)  # play the move
        next_s = canonicalBoard.mirror()  # switch to the other player

        return self.findNext(next_s)  # We continue the descent

    def backpropagation(self, pi, v):
        """
        Update of Qsa and Nsa on every choosen action during the descent

        INPUTS:
            pi (): policy = proba for each move
            v (): evaluation of each state
        """
        if pi != None:  # if policy
            self.Ps[self.boardToPredict.fen()] = pi

        v = -v
        # We go back from final state to the root state
        for (s, a) in self.SAhistory[::-1]:
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                    self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)  # update the score Q
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = v  # update the score Q
                self.Nsa[(s, a)] = 1
            self.Ns[s] += 1  # increase passage throught s
            v = -v

    def search(self, canonicalBoard):
        """
        Descent and backpropagation with Monte Carlo simulation

        INPUT : 
            canonicalBoard () : current state
        """

        if self.selection(canonicalBoard) != None:  # if not final state
            # Evaluation by random descend (pure MCTS)
            v = self.boardToPredict.MonteCarloValue()
            movespi = dict()
            for e in self.boardToPredict.get_legal_moves():
                movespi[e] = 1  # WHY IT IS NOT NORMALIZED ?
            self.backpropagation(movespi, v)
