import os
from tqdm.auto import tqdm
import glob
import chess.pgn

MAX_IMPORT = 100000


def importPgn(filename, s, max_import):
    counter = 0
    total = 0
    with open(filename) as f:
        for line in f:
            if "[Result" in line:
                total += 1
    if total > max_import:
        total = max_import
    pbar = tqdm(total=total, desc="read " + filename,
                unit=" games", mininterval=1)

    pgn = open(filename)
    while counter < max_import:
        # Each time it is called, it reads the next game
        game = chess.pgn.read_game(pgn)
        if not game:
            break
        board = game.board()
        moves = game.mainline_moves()
        count = sum(1 for _ in moves)

        # skip unfinished games
        if count <= 5:
            continue

        result = game.headers["Result"]
        # import only resultative games
        if result != "1-0" and result != "0-1":
            continue

        for move in moves:
            if board.turn == chess.WHITE and result == "1-0":
                line = (
                    "[1-0] "
                    + " ".join(board.fen().split(" ", 2)[:2])
                    + " - "
                    + move.uci()
                ).strip()
                s.add(line)
            elif board.turn == chess.BLACK and result == "0-1":
                line = (
                    "[0-1] "
                    + " ".join(board.fen().split(" ", 2)[:2])
                    + " - "
                    + move.uci()
                ).strip()
                s.add(line)

            board.push(move)

        counter += 1
        pbar.update(1)
    pbar.close()
    return counter


def convert():
    games = 0
    moves = 0
    max_import = MAX_IMPORT
    s = set()

    # load previous state
    if os.path.exists("fen.txt"):
        with open("fen.txt") as f:
            for line in tqdm(f, desc="read fen.txt", unit=" moves", mininterval=1):
                if line:
                    s.add(line)
                    max_import -= 1
                    if max_import <= 0:
                        break

    # Import the new data
    for file in glob.glob("pgn/*.pgn"):
        count = importPgn(file, s, max_import)
        games += count
        max_import -= count
        if max_import <= 0:
            break

    # And we write the data in fen.txt
    with open("fen.txt", "w") as f:
        for line in tqdm(s, desc="write fen.txt", unit=" moves", mininterval=1):
            f.write(line + "\n")
            moves += 1
    print("imported " + str(games) + " games, " + str(moves) + " moves")


if __name__ == "__main__":
    convert()
