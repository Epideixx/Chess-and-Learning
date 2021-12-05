#Format SAN : Qxh7, f6, a8=Q. Format UCI : d2d4, e7e8Q

import chess
import random
from chessboard import display
from time import sleep

board = chess.Board()


display.start(board.fen())
count = 0

def translate_san(coup):
    move = ""
    for i in coup:
        if i == 'D':
            move+='Q'
        elif i == 'R':
            move+='K'
        elif i == 'T':
            move+='R'
        elif i == 'C':
            move+='N'
        elif i == 'F':
            move+='B'
        elif i == '*':
            move+='x'
        else :
            move+= i
    return move

while not board.is_game_over():
    if count % 2 == 0 :
        coup = input()
        board.push_san(translate_san(coup))
        display.update(board.fen())
        sleep(0.2)        
    if count % 2 == 1 :
        legal_count = board.legal_moves.count()
        move_list = list(board.legal_moves)
        which_move = random.randint(0,legal_count-1)
        first_move = move_list[which_move]
        move_holder = chess.Move.from_uci(str(first_move))
        board.push(move_holder)
        display.update(board.fen())
        sleep(0.2)
    count+=1
display.terminate()