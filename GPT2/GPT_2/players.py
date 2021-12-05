from tqdm.auto import tqdm
import chess
from aitextgen.utils import build_gpt2_config
from aitextgen import aitextgen
import os
import random

# Random Player


def random_player(board):
    move = random.choice(list(board.legal_moves))
    return move.uci(), False, False


# GPT-2 Player


model_dir = "trained_model"
vocab_file = "aitextgen-vocab.json"
merges_file = "aitextgen-merges.txt"
max_length = 100

model_dir = "trained_model"
config_file = os.path.join(model_dir, "config.json")
pytorch_model_file = os.path.join(model_dir, "pytorch_model.bin")
vocab_file = os.path.join(model_dir, "aitextgen-vocab.json")
merges_file = os.path.join(model_dir, "aitextgen-merges.txt")
dataset_cache_file = os.path.join(model_dir, "dataset_cache.tar.gz")
max_length = 100

ai = aitextgen(
    model=pytorch_model_file,
    config=config_file,
    vocab_file=vocab_file,
    merges_file=merges_file,
    from_cache=True,
    to_gpu=True,
    # to_fp16=True
)

# a set to find known states
db = set()
with open("fen.txt") as f:
    for line in tqdm(f, desc="read fen.txt", unit=" moves"):
        if line:
            db.add(" ".join(line.split(" ", 3)[:3]))


def gpt2_player(board):
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
    if not move or move not in board.legal_moves:
        # give up and do random move
        move = random.choice(list(board.legal_moves))
        isPredicted = False
    return move.uci(), isPredicted, isKnown


# Human player


def human_player(board):
    uci = get_move("%s's move [q to quit]> " % who(board.turn))
    legal_uci_moves = [move.uci() for move in board.legal_moves]
    while uci not in legal_uci_moves:
        print("Legal moves: " + (",".join(sorted(legal_uci_moves))))
        uci = get_move("%s's move[q to quit]> " % who(board.turn))
    return uci, True, False


def get_move(prompt):
    uci = input(prompt)
    if uci and uci[0] == "q":
        raise KeyboardInterrupt()
    try:
        chess.Move.from_uci(uci)
    except:
        uci = None
    return uci
