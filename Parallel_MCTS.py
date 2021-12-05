import logging
import math
import time
import copy

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class Parallel_MCTS():
    def __init__(self, cpuct, parallel_search):
        self.cpuct = cpuct  # Exploration coefficient
        self.Qsa = {}  # Mean value of the next state ?
        self.Nsa = {}  # Number of time we took action a at state s
        self.Ns = {}   # Number of time we passed through state s
        self.Ps = {}  # Policy
        self.Es = {}  # Value of final states
        self.Ls = {}  # Count the number of leaf node not reached by other descents

        self.parallel_search = parallel_search  # Number of descents
        self.SAhistory = []

    def copy(self):
        """
        Make a deep copy of the parallel MCTS
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
            Boards (dict) : key = int, value = leaf node (board) explored
        """
        self.SAhistories = []  # History of every descents
        self.boardsToPredict = []
        fen = canonicalBoard.fen()  # fen => string representation

        # Here we make our parallel descents
        for _ in range(self.parallel_search):

            # If every leaf node have been reached by other descents ?
            if (fen in self.Ps) and (max([self.Ls[(fen, a)] for a in self.Ps[fen]]) == 0):
                break

            # Else, we make a descent
            self.SAhistory = []
            board = self.findNext(canonicalBoard)

            # If the leaf is not final
            if board != None:
                self.SAhistories.append(self.SAhistory.copy())
                self.boardsToPredict.append(board.copy())

        # We return all boards explored
        return {i: e for (i, e) in enumerate(self.boardsToPredict)}

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
            # +/- 1 if win/lose, 0.01 if equality, 0 else
            self.Es[s] = canonicalBoard.result()

        # IF the node is final
        if self.Es[s] != 0:
            self.backpropagation(
                canonicalBoard, self.SAhistory, None, self.Es[s])
            return None

        # Else, if state s has not been predicted/visited yet
        if s not in self.Ps:
            for (i, j) in self.SAhistory:
                self.Ls[(i, j)] -= 1  # ???
            self.Ns[s] = 0
            return canonicalBoard

        cur_best = -float('inf')
        best_move = -1

        # Else, we continue our descent by choosing the one which maximizes uct
        for a in [e for e in self.Ps[s] if self.Ls[(s, e)] > 0]:

            # If action a has already been choosen once
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * \
                    math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])

            # Else, if a is choosen for the first time
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

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

    def backpropagation(self, board, history, pi, v):
        """
        Mise à jour des Qsa, Nsa et Ls de toutes les actions choisies lors d'une descente
        ==> TO CHANGE

        """
        newNodes = 0
        if pi != None:  # if policy
            fen = board.fen()
            self.Ps[fen] = pi
            newNodes = len(pi)
            for a in pi:
                self.Ls[(fen, a)] = 1

        v = -v
        # We go back from final state to the root state
        for (s, a) in history[::-1]:
            self.Ls[(s, a)] += newNodes
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                    self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)  # update the score Q
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = v  # update the score Q
                self.Nsa[(s, a)] = 1
            self.Ns[s] += 1  # increase passage throught s
            v = -v

    def parallel_backpropagation(self, pi, v):
        """
        Mise à jour des Qsa, Nsa et Ls de toutes les descentes
        """
        for i in range(len(self.boardsToPredict)):
            self.backpropagation(
                self.boardsToPredict[i], self.SAhistories[i], pi[i], v[i])
