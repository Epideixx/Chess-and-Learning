import time
from IPython.display import display, HTML, clear_output
import chess


def who(player):
    return "White" if player == chess.WHITE else "Black"


def display_board(board, use_svg):
    if use_svg:
        return board._repr_svg_()
    else:
        return "<pre>" + str(board) + "</pre>"


def play_game(player1, player2, visual="svg", pause=0.1):
    """
    playerN1, player2: functions that takes board, return uci move
    visual: "simple" | "svg" | None
    """
    use_svg = (visual == "svg")
    board = chess.Board()
    known1 = 0
    predicted1 = 0
    total1 = 0
    known2 = 0
    predicted2 = 0
    total2 = 0
    if visual is not None:
        display(display_board(board, visual == 'svg'))
    try:
        while not board.is_game_over(claim_draw=True):
            if board.turn == chess.WHITE:
                uci, isPredicted, isKnown = player1(board)
                total1 += 1
                if isKnown:
                    known1 += 1
                if isPredicted:
                    predicted1 += 1
            else:
                uci, isPredicted, isKnown = player2(board)
                total2 += 1
                if isKnown:
                    known2 += 1
                if isPredicted:
                    predicted2 += 1
            name = who(board.turn)
            board.push_uci(uci)
            board_stop = display_board(board, use_svg)

            html = "<b>Move %s %s, Play '%s':</b><br/>%s<br/>Known/Predicted/Total moves: %s/%s/%s %s%% - %s/%s/%s %s%%" % (
                len(board.move_stack), name, uci, board_stop,
                known1, predicted1, total1, round(
                    predicted1 / (total1 or 1) * 100),
                known2, predicted2, total2, round(predicted2 / (total2 or 1) * 100))
            if visual is not None:
                if visual == "svg":
                    clear_output(wait=True)
                display(HTML(html))
                if visual == "svg":
                    time.sleep(pause)
    except KeyboardInterrupt:
        msg = "Game interrupted!"
        return (None, msg, board)
    result = "1/2-1/2"
    if board.is_checkmate():
        msg = "checkmate: " + who(not board.turn) + " wins!"
        result = "1-0" if who(not board.turn) == "White" else "0-1"
    elif board.is_stalemate():
        msg = "draw: stalemate"
    elif board.is_fivefold_repetition():
        msg = "draw: 5-fold repetition"
    elif board.is_insufficient_material():
        msg = "draw: insufficient material"
    elif board.can_claim_draw():
        msg = "draw: claim"
    if visual is not None:
        print(msg)
    return (result, msg, board)
