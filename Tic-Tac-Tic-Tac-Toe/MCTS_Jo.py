# -------------------------------------------------
#           Monte Carlo Tree Search
# -------------------------------------------------

from math import sqrt
import numpy as np
import game
from copy import deepcopy
from random import randint, choice


# La priorité est donnée par l'IA d'évaluation de la stratégie
# La valeur win de chance de gagner est donnée par l'IA d'évaluation de victoire


class Node:

    def __init__(self, player, priority, tictactictactoe):
        self.nb_visit = 0
        self.win = 0
        self.q = 0
        self.prior = priority  # Donné par IA

        self.to_play = player
        self.children = {}
        # key = node
        # value = action
        self.expanded = False
        self.state = tictactictactoe

    def expand(self):
        '''
        Feuille que l'on souhaite étendre si cela est possible
        '''

        coups_possibles = self.state.all_moves()
        n = len(coups_possibles)

        for coup in coups_possibles:
            position_child = deepcopy(self.state)

            position_child.play(coup, self.to_play)

            # Le 1/n correspond normalement à la valeur donnée par l'IA de stratégie
            if self.to_play == "O":
                child = Node("X", 1/n, position_child)
            else:
                child = Node("O", 1/n, position_child)

            self.children[child] = coup
            self.expanded = True


def select_rec(parent, path):
    '''
    On va partir de la racine, et étendre en utilisant UCB pour descendre jusqu'à une racine
    Et retourne le noeud sélectionné ainsi que le chemin pour y accéder
    '''
    best_score = - np.inf
    best_child = None

    if parent.children == {}:
        return (parent, path)

    for child in parent.children.keys():
        score = uct(parent, child, parent.to_play)
        if score > best_score:
            best_score = score
            best_child = child

    return select_rec(best_child, path + [best_child])


def backpropagation(path, node_selected, result):
    '''
    Met à jour toutes les noeuds sur le chemin suivi
    '''

    for node in path:
        node.nb_visit += 1
        if node.to_play == "X":  # Du coup pour X il faut maximiser le score car victoire +1
            node.win += result
        else:
            node.win += result  # Et pour O il faut minimiser car victoire -1

        node.q = node.win/node.nb_visit


def uct(parent, child, player):
    '''
    Score d'une arête reliant un noeud parent à un noeud fils
    '''
    if player == "X":
        k = 1
    else:
        k = -1

    if child.nb_visit == 0:
        return 1000
    c = 1/len(parent.children)
    return k*child.q + sqrt((parent.nb_visit)/(child.nb_visit))
    # On met un moins parce qu'on passe au joueur opposé donc on veut le score opposé


class MCTS:

    def __init__(self, firstplayer, iteration):
        self.iteration = iteration
        self.firstplayer = firstplayer
        self.player = firstplayer
        self.noderoot = Node(self.player, 0, game.TicTacTicTacToe())
        self.path = [self.noderoot]
        self.path_coups = []

    def run(self):
        # Création d'un noeud root pour début du jeu
        root = self.noderoot
        for _ in range(1, self.iteration + 1):
            # Descente dans l'arbre des noeuds jusqu'à une feuille
            node, path = select_rec(root, [root])

            fin = node.state.representation.status()

            # Cas où la feuille est finale
            if fin in ["X", "O", "T"]:
                if fin == "X":
                    eval_node = 1
                elif fin == "O":
                    eval_node = -1
                else:
                    eval_node = 0
                backpropagation(path, node, eval_node)

            # Cas où la feuille n'est pas terminale
            else:  # Modif ici
                eval_node = jeuAleatoire(node.state, node.to_play)
                backpropagation(path, node, eval_node)
                # Extension de la feuille
                node.expand()

    def bestcoup(self):
        self.run()

        best_score = - np.inf
        best_coup = None

        for node in list(self.noderoot.children.keys()):
            if best_score < node.nb_visit:
                best_score = node.nb_visit
                best_coup = self.noderoot.children[node]

        return best_coup

    def update(self, coup):

        liste = [
            node for (node, c) in self.noderoot.children.items() if c == coup]
        self.noderoot = liste[0]
        if self.player == "X":
            self.player = "O"
        else:
            self.player = "X"
        self.path.append(self.noderoot)
        self.path_coups.append(coup)

    def examples(self, iterations, convertdata_function):
        """
        Retourne tous les plateaux de jeu qui ont mené à la fin du jeu, la victoire/défaite associée
        """

        res = []

        for _ in range(iterations):

            # Initialisation
            self.player = self.firstplayer
            self.noderoot = Node(self.player, 0, game.TicTacTicTacToe())
            self.path = [self.noderoot]
            self.path_coups = []

            while True:

                a = self.bestcoup()
                self.update(a)

                jeu = self.noderoot.state

                # ON ARRETE SI LE JEU EST FINI
                if jeu.representation.status() in ["O", "X", "T"]:

                    if jeu.representation.status() == "X":
                        res = res + \
                            convertdata_function(self.path, self.path_coups, 1)
                    elif jeu.representation.status() == "O":
                        res = res + \
                            convertdata_function(
                                self.path, self.path_coups, -1)
                    else:
                        res = res + \
                            convertdata_function(self.path, self.path_coups, 0)
                    break

                possible_moves = jeu.all_moves()

                a = choice(possible_moves)

                self.update(a)

                jeu = self.noderoot.state

                # ON ARRETE SI LE JEU EST FINI
                if jeu.representation.status() in ["O", "X", "T"]:

                    if jeu.representation.status() == "X":
                        res = res + \
                            convertdata_function(self.path, self.path_coups, 1)
                    elif jeu.representation.status() == "O":
                        res = res + \
                            convertdata_function(
                                self.path, self.path_coups, -1)
                    else:
                        res = res + \
                            convertdata_function(self.path, self.path_coups, 0)
                    break
        return res


def convertdata(path, path_coups, score):
    """
    A partir d'une suite de noeuds et du score final, renvoie des données utiles pour le réseau neuronal
    A MODIFIER POUR LES COUPS !!!
    """

    res = []
    for node in path:
        plateau = np.zeros((9, 9, 3))
        representation = node.state.board
        for i in range(9):
            for j in range(9):
                a = representation[9*i + j]
                if a == "X":
                    plateau[i, j, 0] = 1
                    plateau[i, j, 1] = 0
                    plateau[i, j, 2] = 0
                elif a == "O":
                    plateau[i, j, 0] = 0
                    plateau[i, j, 1] = 1
                    plateau[i, j, 2] = 0
                else:
                    plateau[i, j, 0] = 0
                    plateau[i, j, 1] = 0
                    plateau[i, j, 2] = 1
        # On effectue aussi une transformatin du score de [-1; 1] vers [0, 1]
        res.append((plateau, np.array([((1+score)/2.1)])))

    return res


def convertdata_dense(path, path_coups, score):
    """
    A partir d'une suite de noeuds et du score final, renvoie des données utiles pour le réseau neuronal
    """
    res = []
    for i in range(len(path_coups)):
        coups = np.zeros(81)
        node = path[i]
        coup = path_coups[i]
        coups[coup] = 1
        res.append((node.state.representation_reseau(
            node.to_play), (1+score)/2, coups))

    return res


def jeuAleatoire(tictactictactoe, to_play):
    """
    A partir d'une position de Jeu, effectue une partie aléatoire et retourne le résultat
    """
    game = tictactictactoe.copy()

    turn = 0
    if to_play == "X":
        symbols = ["X", "O"]
    else:
        symbols = ["O", "X"]

    end = ' '
    while not(end in ["O", "X", "T"]):

        symbol = symbols[turn]

        possible_moves = game.all_moves()

        a = possible_moves[np.random.randint(len(possible_moves))]

        game.play(a, symbol)

        end = game.representation.status()

        turn = 1 - turn

    winner = end

    # Rappel: +1 pour X et -1 pour O

    if winner == "X":
        return 1
    elif winner == "O":
        return -1
    elif winner == "T":
        return 0
    else:
        raise ValueError


# Phase de test
if __name__ == '__main__':
    mcts = MCTS("O", 2000)
    mcts.run()
    print(mcts.noderoot.q)
    print(mcts.noderoot.win)
    print(mcts.noderoot.nb_visit)
    print("Done")
    print(mcts.examples(1, convertdata_dense))
