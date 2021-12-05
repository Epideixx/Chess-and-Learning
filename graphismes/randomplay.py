import chess
import random
from chessboard import display
from time import sleep

board = chess.Board()


display.start(board.fen())
while not board.is_game_over():
    legal_count = board.legal_moves.count()
    move_list = list(board.legal_moves)
    print(move_list)
    which_move = random.randint(0,legal_count-1)
    print(which_move)
    first_move = move_list[which_move]
    print(first_move)
    move_holder = chess.Move.from_uci(str(first_move))
    print(move_holder)
    board.push(move_holder)
    display.update(board.fen())
    sleep(0.2)
display.terminate()