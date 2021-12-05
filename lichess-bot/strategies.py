"""
Some example strategies for people who want to create a custom, homemade bot.
And some handy classes to extend
"""

import chess
from chess.engine import PlayResult
import random
from engine_wrapper import EngineWrapper

from tqdm.auto import tqdm
from aitextgen.utils import build_gpt2_config
from aitextgen import aitextgen
import os


class FillerEngine:
    """
    Not meant to be an actual engine.

    This is only used to provide the property "self.engine"
    in "MinimalEngine" which extends "EngineWrapper"
    """

    def __init__(self, main_engine, name=None):
        self.id = {"name": name}
        self.name = name
        self.main_engine = main_engine

    def __getattr__(self, method_name):
        main_engine = self.main_engine

        def method(*args, **kwargs):
            nonlocal main_engine
            nonlocal method_name
            return main_engine.notify(method_name, *args, **kwargs)

        return method


class MinimalEngine(EngineWrapper):
    """
    Subclass this to prevent a few random errors

    Even though MinimalEngine extends EngineWrapper,
    you don't have to actually wrap an engine.

    At minimum, just implement `search`,
    however you can also change other methods like
    `notify`, `first_search`, `get_time_control`, etc.
    """

    def __init__(self, *args, name=None):
        super().__init__(*args)

        self.engine_name = self.__class__.__name__ if name is None else name

        self.last_move_info = []
        self.engine = FillerEngine(self, name=self.name)
        self.engine.id = {"name": self.engine_name}

    def search_with_ponder(self, board, wtime, btime, winc, binc, ponder, draw_offered):
        timeleft = 0
        if board.turn:
            timeleft = wtime
        else:
            timeleft = btime
        return self.search(board, timeleft, ponder, draw_offered)

    def search(self, board, timeleft, ponder, draw_offered):
        """
        The method to be implemented in your homemade engine

        NOTE: This method must return an instance of "chess.engine.PlayResult"
        """
        raise NotImplementedError("The search method is not implemented")

    def notify(self, method_name, *args, **kwargs):
        """
        The EngineWrapper class sometimes calls methods on "self.engine".
        "self.engine" is a filler property that notifies <self>
        whenever an attribute is called.

        Nothing happens unless the main engine does something.

        Simply put, the following code is equivalent
        self.engine.<method_name>(<*args>, <**kwargs>)
        self.notify(<method_name>, <*args>, <**kwargs>)
        """
        pass


class ExampleEngine(MinimalEngine):
    pass


# Strategy names and ideas from tom7's excellent eloWorld video


class RandomMove(ExampleEngine):
    def search(self, board, *args):
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


# GPT-2 Player

current_working_directory = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "GPT2", "GPT_2"
)
os.chdir(current_working_directory)
model_dir = "trained_model"
vocab_file = "aitextgen-vocab.json"
merges_file = "aitextgen-merges.txt"
max_length = 100

config_file = os.path.join(model_dir, "config.json")
pytorch_model_file = os.path.join(model_dir, "pytorch_model.bin")
vocab_file = os.path.join(model_dir, "aitextgen-vocab.json")
merges_file = os.path.join(model_dir, "aitextgen-merges.txt")
dataset_cache_file = os.path.join(model_dir, "dataset_cache.tar.gz")
max_length = 100

ai = aitextgen(
    model_folder=model_dir,
    config=config_file,
    vocab_file=vocab_file,
    merges_file=merges_file,
    from_cache=True,
    to_gpu=False,
    # to_fp16=True
)

# a set to find known states
db = set()
with open("fen.txt") as f:
    for line in tqdm(f, desc="read fen.txt", unit=" moves"):
        if line:
            db.add(" ".join(line.split(" ", 3)[:3]))


class GPT2Move(ExampleEngine):
    def search(self, board, *args):
        if board.turn == chess.WHITE:
            prompt = "[1-0] " + " ".join(board.fen().split(" ", 2)[:2])
        else:
            prompt = "[0-1] " + " ".join(board.fen().split(" ", 2)[:2])
        isKnown = prompt in db

        prediction = ai.generate_one(
            prompt=prompt,
            max_length=max_length,
            temperature=0.9,
            top_k=0,
        )
        isPredicted = False
        try:
            uci = prediction.split(" - ")[1].strip()
            move = chess.Move.from_uci(uci)
            isPredicted = True
        except Exception as e:
            # print(str(e))
            move = None
            print(move)
        if not move or move not in board.legal_moves:
            # give up and do random move
            move = random.choice(list(board.legal_moves))
            isPredicted = False
            print("Dommage, on joue le hasard")
        return PlayResult(move, None)


class FirstMove(ExampleEngine):
    """Gets the first move when sorted by uci representation"""

    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)
