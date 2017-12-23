import chess
from chess import svg
import cairosvg
from chess import pgn
from io import StringIO
import random
import sys

import xml.etree.ElementTree as ET

FOLDERS = {
	0: 'train',
	1: 'test',
	2: 'extra'
}

def fen_to_png(fen):
    board = chess.Board("%s w - - 0 1" % fen)

    svg = chess.svg.board(board=board, size=100)

    filename = fen.replace("/", "-")

    folder = FOLDERS[random.randint(0, 2)]
    cairosvg.svg2png(bytestring=svg, write_to="./data/%s/%s.png" % (folder, filename))
    #print("Wrote %s.png" % filename)

raw = open("./data/fens/Adams.pgn", "r").read()
lines = [line for line in raw.split("\n") if line.count("[") == 0]
processed = "\n".join(lines)
games = processed.split("\n\n")

fens = {}
for game_str in games:
    pgn = StringIO(game_str)
    game = chess.pgn.read_game(pgn)
    
    node = game
    try:
        while not node.is_end():
            next_node = node.variation(0)
            fen = node.board().fen().split(" ")[0]
            if fen not in fens:
                fens[fen] = True
                fen_to_png(fen)
            node.board().san(next_node.move)
            node = next_node
    except Exception as e:
			  print(e)
