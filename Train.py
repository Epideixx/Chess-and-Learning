import copy
from tqdm import tqdm  # Loading bar
import numpy as np
from random import shuffle
from MCTS import MCTS
from Arena import Arena
from Models import NeuralNetwork
# Enable to get some information when getting some error, or debugging...
import logging
import time
import coloredlogs  # Put some colors in log messages
import pickle  # I DON'T KNOW HOW TO EXPLAIN ...
import wandb

from Games import Connect4 as Game

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = {
<<<<<<< HEAD
    'numIters': 200,  # ?
    'numEps': 256,  # ?
    'numParallelGame': 256,  # Number of games in parallel
    'numMCTSSims': 20,  # Number of descents in the MCTS
    'arenaBatch': 1,  # Number of batch of parallel games for Arena
    'numParallelGameArena': 50,  # Number of games played in parallel (Arena)
    'cpuct': 1,  # Exploration coefficient
=======
    'numIters': 200,
    'numEps': 256,
    'numParallelGame': 256,
    'numMCTSSims': 20,
    'arenaBatch': 1,
    'numParallelGameArena': 50,
    'cpuct': 1,
>>>>>>> master

    'checkpoint': './data/history',
    'resume_wandb': False,
    'resume_model_and_data': False,
    'warm_start': False,
    'resume_iteration': 0,
    'model_file_name': 'best.h5',
}


def main():
    """
    Main function for test
    """
    train()


def generate_data(NN):
<<<<<<< HEAD
    """
    Generate data by playing games and returns it in order to be used to train the NN

    INPUT:
        NN (NeuralNetwork) : Neural network which will be used to evaluate pi and v for each state of the MCTS

    OUTPUT:
        iterationTrainExamples (list) : List of the form (board, pi, result) for each board visited
    """
    # Initialization of the items
    iterationTrainExamples = []
    pbar = tqdm(total=args['numEps'], desc="Self Play")  # Loading bar
    boards = []
    mcts = []
    trainExamples = []  # Data for training
    ended = [0]*args['numParallelGame']  # Monitoring of the end of games
=======
    # Initialisation des variables
    iterationTrainExamples = []
    pbar = tqdm(total=args['numEps'], desc="Self Play")
    boards = []
    mcts = []
    trainExamples = []
    ended = [0]*args['numParallelGame']
>>>>>>> master
    currentPlayer = [0]*args['numParallelGame']
    episodeStep = [0]*args['numParallelGame']  # Number of played moves
    nbrGameStarted = args['numParallelGame']  # Number of games running
    for i in range(args['numParallelGame']):
<<<<<<< HEAD
        boards.append(Game())  # Initializatino of the boards
        # Initialization of the MCTS
        mcts.append([MCTS(args['cpuct']), MCTS(args['cpuct'])])
        trainExamples.append([])

    # While a game is still running
    while not min(ended):

        for i in range(args['numParallelGame']):
            episodeStep[i] += 1

        # One loop per simulation
        for _ in range(args['numMCTSSims']):
            # We initialize a dictionnary which will receive the state games to predict with the NN
            boardsToPredict = dict()

            # For every game which is not terminated
            for i in [k for k in range(args['numParallelGame']) if not ended[k]]:
                # We make a descent for each MCTS to find a state to predict
                boardToPredict = mcts[i][currentPlayer[i]].selection(boards[i])
                if boardToPredict != None:  # If not a final state
                    boardsToPredict[i] = boardToPredict

            if len(boardsToPredict) > 0:
                # We evaluate pi and v for every board
                pi, v = NN.predictBatch(boardsToPredict)

                # Then, we update the MCTS
=======
        boards.append(Game())
        mcts.append([MCTS(args['cpuct']), MCTS(args['cpuct'])])
        trainExamples.append([])

    while not min(ended):  # Boucle tant qu'un des processus n'est pas terminé

        for i in range(args['numParallelGame']):
            episodeStep[i] += 1

        for _ in range(args['numMCTSSims']):  # Un tour de boucle par simulation
            # On initialise un dictionnaire auquel on va ajouter tous les etats de jeu à predict avec le réseau de neurone
            boardsToPredict = dict()
            for i in [k for k in range(args['numParallelGame']) if not ended[k]]:
                # On fait une descente dans chaque mcts pour trouvé un état de jeu à predict
                boardToPredict = mcts[i][currentPlayer[i]].selection(boards[i])
                if boardToPredict != None:
                    boardsToPredict[i] = boardToPredict
            if len(boardsToPredict) > 0:
                # On estime pi et v pour tous les jeux puis on met à jour les mcts
                pi, v = NN.predictBatch(boardsToPredict)
>>>>>>> master
                for i in boardsToPredict:
                    mcts[i][currentPlayer[i]].backpropagation(pi[i], v[i])

        # For every game which is not terminated
        for i in [k for k in range(args['numParallelGame']) if not ended[k]]:
<<<<<<< HEAD

            if False and i % 2 == 0 and episodeStep[i] < 6:  # ???????
=======
            if False and i % 2 == 0 and episodeStep[i] < 6:
>>>>>>> master
                boards[i].playRandomMove()

            else:
                pi = mcts[i][currentPlayer[i]].getActionProb(boards[i], temp=(
<<<<<<< HEAD
                    1 if episodeStep[i] < 20 else 0))  # Action probability
                # temp = 1 ==> proba else temp = 0 ==> only the best action

                # We save board state and its symetries (to get more data !!!)
                for sym, pis in boards[i].get_symmetries(pi):
                    trainExamples[i].append((sym, pis, currentPlayer[i]))
                # sym = board represention of the symmetry; pis = pi of the symmetry

                # Selection of the next move according to the prob
=======
                    1 if episodeStep[i] < 20 else 0))  # probabilité d'action
                # On sauvegarde l'état du jeu et ses symétries
                for sym, pis in boards[i].get_symmetries(pi):
                    trainExamples[i].append((sym, pis, currentPlayer[i]))
                # trainExamples[i].append((boards[i].representation,pi,currentPlayer[i]))

                # Selection d'un coup à jouer
>>>>>>> master
                d = {i: e for i, e in enumerate(list(pi.keys()))}
                move = np.random.choice(list(d.keys()), p=list(pi.values()))
                move = d[move]

<<<<<<< HEAD
                # Let's play the move
                boards[i].push(move)

            # If the i-th game is over, add data to training data
            if boards[i].is_game_over():
                r = boards[i].result()
                iterationTrainExamples += [(x[0], x[1], np.round(r) * (
                    (-1) ** (x[2] != currentPlayer[i]))) for (k, x) in enumerate(trainExamples[i])]  # (board, pi, result)

                # If there are other games to play, we initialize the variables
=======
                # On joue le coup
                boards[i].push(move)

            # Si la partie est terminée, ajoute des données aux données d'entraînement
            if boards[i].is_game_over():
                r = boards[i].result()
                iterationTrainExamples += [(x[0], x[1], np.round(r) * (
                    (-1) ** (x[2] != currentPlayer[i]))) for (k, x) in enumerate(trainExamples[i])]

                # Si il y a d'autres partie à jouer, on réinitialise les variables
>>>>>>> master
                if nbrGameStarted < args['numEps']:
                    nbrGameStarted += 1
                    currentPlayer[i] = 0
                    episodeStep[i] = 0
                    boards[i] = Game()
                    mcts[i] = [MCTS(args['cpuct']), MCTS(args['cpuct'])]
                    trainExamples[i] = []
<<<<<<< HEAD
                # Else, it is the end of this game
                else:
                    ended[i] = 1
                pbar.update(1)

            # Else the game is not terminated, so let's inverse the board, and continue playing
            else:
                boards[i] = boards[i].mirror()
                currentPlayer[i] = (currentPlayer[i]+1) % 2

=======
                else:
                    ended[i] = 1
                pbar.update(1)
            else:  # Sinon on inverse le plateau et on change de joueur
                boards[i] = boards[i].mirror()
                currentPlayer[i] = (currentPlayer[i]+1) % 2
>>>>>>> master
    pbar.close()
    return iterationTrainExamples


<<<<<<< HEAD
def train(loss_weights=[1, 1, 1, 1]):
    """
    Train a neural network and save it        
    """

=======
def train():
>>>>>>> master
    log.info('START OF TRAINING IN 5 SECONDS...')
    time.sleep(5)

    log.info('Initialization')
<<<<<<< HEAD
    # Initialization of the Neural Network
=======
    # Initialisation du réseau de neurones et chargement des poids
>>>>>>> master
    NN = NeuralNetwork()
    # If exists, load a model
    if args['resume_model_and_data']:
        log.info('Loading model...')
        NN.load_checkpoint(
            folder=args['checkpoint'], filename=args['model_file_name'])
<<<<<<< HEAD
        NN.compile(loss_weights)
=======
        NN.compile()
>>>>>>> master
        log.info('Model succesfully loaded')
    NN.save_checkpoint(folder=args['checkpoint'], filename='best.h5')
    NN.save_checkpoint(folder=args['checkpoint'], filename='save0.h5')

<<<<<<< HEAD
    # We fetch the data
=======
    # Récupération des données
>>>>>>> master
    if args['warm_start']:
        log.info('Loading warm start data...')
        data_file = open("data/warm_start_data.plk", "rb")
        trainExamplesHistory = pickle.load(data_file)
        data_file.close()
        log.info('Data succesfully loaded')
        trainExamples = []
        for e in trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)
        log.info('Starting warm start')
        # We train the NN with trainExamples
        NN.train(trainExamples, warm_start=True)
    else:
        if args['resume_model_and_data']:
            log.info('Loading data...')
            data_file = open("data/data.plk", "rb")
            trainExamplesHistory = pickle.load(data_file)
            data_file.close()
            log.info('Data succesfully loaded')
        else:
            trainExamplesHistory = []

<<<<<<< HEAD
    # Tracker activation ???
    NN.start_wandb(args['resume_wandb'])

    # Let's begin the training !!!

=======
    # Activation du tracker
    NN.start_wandb(args['resume_wandb'])

    # Début de l'entrainement
>>>>>>> master
    if args['resume_model_and_data']:
        start_iter = args['resume_iteration']
    else:
        start_iter = 0
<<<<<<< HEAD

    # Let's make trainings
    for iteration in range(start_iter, args['numIters']):
        log.info(f'Iteration #{iteration}')

        # Every iteration makes play "numParallelGame" games in parallel
        # We generate data
        iterationTrainExamples = generate_data(NN)

        # In order to only train the model with good data, we only keep the "limit" last series of data
=======
    for iteration in range(start_iter, args['numIters']):
        log.info(f'Iteration #{iteration}')

        # Chaque iteration fait jouer 'numParallelGame' parties en parallèle
        iterationTrainExamples = generate_data(NN)

        # Limitation du nombre de données d'entrainement
>>>>>>> master
        if args['warm_start']:
            limit = 20
        else:
            limit = max(5, min(20, 3+iteration//2))
<<<<<<< HEAD
        # Add new data
        trainExamplesHistory.append(iterationTrainExamples)
        # If too much data, we delete the first data
        while len(trainExamplesHistory) > limit:
            trainExamplesHistory.pop(0)

        # We save training data
=======
        trainExamplesHistory.append(iterationTrainExamples)
        while len(trainExamplesHistory) > limit:
            trainExamplesHistory.pop(0)

        # Sauvegarde des données d'entrainement
>>>>>>> master
        data_file = open("data/data.plk", "wb")
        pickle.dump(trainExamplesHistory, data_file)
        data_file.close()

<<<<<<< HEAD
        # Let's create the train dataset
=======
        # Création du dataset de train
>>>>>>> master
        trainExamples = []
        for e in trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        NN.save_checkpoint(folder=args['checkpoint'], filename='temp.h5')
        lastNN = NeuralNetwork()
        # We take the previous NN
        lastNN.load_checkpoint(folder=args['checkpoint'], filename='temp.h5')
        NN.train(trainExamples)  # We train the new NN

<<<<<<< HEAD
        # Let's compare the new NN to a simple MCTS thanks to Arena
=======
>>>>>>> master
        arena = Arena(Game, [NN, MCTS(args['cpuct'])],
                      ["mcts", MCTS(args['cpuct'])])
        mctswinsNew, mctswinsLast, mctsdraw = arena.compare(args)
        log.info('New wins : %d ; Mcts wins %d ; Draws : %d' %
                 (mctswinsNew, mctswinsLast, mctsdraw))

<<<<<<< HEAD
        # Let's compare the two NN thanks to the Arena
=======
>>>>>>> master
        arena = Arena(Game, [NN, MCTS(args['cpuct'])],
                      [lastNN, MCTS(args['cpuct'])])
        winsNew, winsLast, draw = arena.compare(args)
        log.info('New wins : %d ; Previous wins %d ; Draws : %d' %
                 (winsNew, winsLast, draw))

        # DON'T KNOW HOW EXACTLY IT WORKS BUT IT SEEMS TO SAVE THE RESULTS
        wandb.log({
            "window_size": len(trainExamplesHistory),
            "wins_against_mcts": mctswinsNew,
            "losses_against_mcts": mctswinsLast,
            "draws_against_mcts": mctsdraw,
            "wins_against_self": winsNew,
            "losses_against_self": winsLast,
            "draws_against_self": draw
        }, commit=False)

<<<<<<< HEAD
        # We keep the best model of both
=======
>>>>>>> master
        if winsNew > winsLast:
            log.info('ACCEPTING NEW MODEL')
            NN.save_checkpoint(folder=args['checkpoint'], filename='best.h5')
            NN.save_checkpoint(
                folder=args['checkpoint'], filename=f'save{iteration+1}.h5')
        else:
            log.info('REJECTING NEW MODEL')
            NN.load_checkpoint(folder=args['checkpoint'], filename='best.h5')


if __name__ == "__main__":
    main()
