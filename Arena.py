# Enable to get some information when getting some error, or debugging...
import logging

import numpy as np
from tqdm import tqdm  # Enable to have a progress bar while loading
from MCTS import MCTS
from Parallel_MCTS import Parallel_MCTS

log = logging.getLogger(__name__)  # Create the logger object


class Arena():
    def __init__(self, Game, player1, player2):
        """
        Initialize the game.

        INPUTS :
            Game (Games) : Game object from Games.py
            player1 (list with two objects) : Player with the shape [PredictionMaker = NN or MCTS, DecisionMaker = MCTS]
            player2 (list with two objects) : Player with the shape [PredictionnMaker = NN or MCTS, DecisionMaker = MCTS] 

        """
        self.GameClass = Game
        self.player1 = player1
        self.player2 = player2

    def start(self):
        """
        Main function which makes games and returns the results

        OUTPUT :
            results (list) : list of 1 if player 1 wins, -1 if player 2 wins, 0 else, depending of the game
        """

        # Note : self.compare is called before self.start !

        players = [self.player1, self.player2]
        pbar = tqdm(total=self.args['arenaBatch'] *
                    self.args['numParallelGameArena'], desc="Arena")
        # opening of the progress bar
        # total : "number of expected iteration"
        # desc : "description"

        results = []

        # Each batch of games (played in parallel)
        for _ in range(self.args['arenaBatch']):
            boards = []
            mcts = []

            # Follows if each game of the batch is terminated or not
            ended = [0]*self.args['numParallelGameArena']
            currentPlayer = 0
            episodeStep = 0  # Progression of the games

            # Preparation for games in parallel
            for i in range(self.args['numParallelGameArena']):
                boards.append(self.GameClass())
                mcts.append([self.player1[1].copy(), self.player2[1].copy()])

            while not min(ended):
                episodeStep += 1
                if players[currentPlayer][0] != None:  # if there is a NN or a simple mcts

                    # MCTS descents
                    for _ in range(self.args['numMCTSSims'][currentPlayer]):

                        # MCTS player
                        if players[currentPlayer][0] == "mcts":
                            for i in [k for k in range(self.args['numParallelGameArena']) if not ended[k]]:
                                mcts[i][currentPlayer].search(boards[i])

                        # Neural Network player + MCTS
                        else:
                            # Simple MCTS
                            if isinstance(players[currentPlayer][1], MCTS):  # ???
                                boardsToPredict = dict()

                                # Create list of current boards for each game still running
                                for i in [k for k in range(self.args['numParallelGameArena']) if not ended[k]]:
                                    boardToPredict = mcts[i][currentPlayer].selection(
                                        boards[i])
                                    if boardToPredict != None:
                                        boardsToPredict[i] = boardToPredict

                                # Prediction of the policy and evaluation of the state by the NN (if it exists)
                                if len(boardsToPredict) > 0:
                                    # "pi" = policy
                                    # "v" = evaluation of the state
                                    pi, v = players[currentPlayer][0].predictBatch(
                                        boardsToPredict)
                                    for i in boardsToPredict:
                                        # Backpropagation of the stats on the MCTS
                                        mcts[i][currentPlayer].backpropagation(
                                            pi[i], v[i])

                            # Parellel MCTS ==> ???
                            else:
                                for i in [k for k in range(self.args['numParallelGameArena']) if not ended[k]]:
                                    boardsToPredict = mcts[i][currentPlayer].selection(
                                        boards[i])
                                    if len(boardsToPredict) > 0:
                                        pi, v = players[currentPlayer][0].predictBatch(
                                            boardsToPredict)
                                        mcts[i][currentPlayer].parallel_backpropagation(
                                            pi, v)

                for i in [k for k in range(self.args['numParallelGameArena']) if not ended[k]]:
                    if players[currentPlayer][0] != None:

                        # Gets the probability of each possible move
                        pi = mcts[i][currentPlayer].getActionProb(
                            boards[i], temp=(2 if episodeStep < 10 else 0))  # temp ???
                        d = {i: e for i, e in enumerate(list(pi.keys()))}

                        # We choose the next based on the probability
                        move = np.random.choice(
                            list(d.keys()), p=list(pi.values()))
                        move = d[move]

                        # We play the move
                        boards[i].push(move)

                    # If no MCTS or NN ==> Random player
                    else:
                        boards[i].playRandomMove()

                    r = boards[i].result()

                    # If the game is ended by the victory of one of the player
                    if r != 0:
                        results.append(r*(-1)**currentPlayer)
                        ended[i] = 1
                        pbar.update(1)
                    # Else we just switch the board
                    else:
                        boards[i] = boards[i].mirror()

                # And we change the player
                currentPlayer = (currentPlayer+1) % 2
        pbar.close()

        return results

    def compare(self, args, verbose=False):
        """
        Returns the results of the two players in order to get the best version

        INPUTS:
            args (dict) : 
                - 'numMCTSSims': [int1, int2] or int=> number of MCTS simulation
                - 'arenaBatch': int => number of batch of games
                - 'numParallelGameArena': int => number of games per batch
                - 'cpuct' : [int1, int2] or int=> ?

        OUTPUT:
            winNew, winLast, draw (tuple) : wins of the new player, of the second player and equality
        """
        self.args = args.copy()

        # If numMCTSSims or cpuct is just an int, let's change it into a list with twice the same int
        if type(self.args['numMCTSSims']) == int:
            self.args['numMCTSSims'] = [
                self.args['numMCTSSims'], self.args['numMCTSSims']]
        if type(self.args['cpuct']) == int:
            self.args['cpuct'] = [self.args['cpuct'], self.args['cpuct']]

        winNew = 0  # Victory of the first player
        winLast = 0  # Victory of the second player
        draw = 0  # Equality

        results = self.start()  # Player 1 plays first
        for r in results:
            if r == 1:
                winNew += 1
            elif r == -1:
                winLast += 1
            else:
                draw += 1

        # Switch player1 and player2
        self.player1, self.player2 = self.player2, self.player1
        self.args['numMCTSSims'] = self.args['numMCTSSims'][::-1]

        results = self.start()  # Player 2 plays first
        for r in results:
            if r == 1:
                winLast += 1
            elif r == -1:
                winNew += 1
            else:
                draw += 1

        return winNew, winLast, draw
