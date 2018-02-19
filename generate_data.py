import chess
from chess import svg
import cairosvg
from chess import pgn
from io import StringIO
import random
import sys
import glob

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

    r = random.randint(0, 24)
    if r < 3:
        x = 0
    elif r < 6:
        x = 1
    else:
        x = 2
    folder = FOLDERS[x]
    cairosvg.svg2png(bytestring=svg, write_to="./out_of_sample_data/%s.png" % (filename))

def clean(pgn_file):
    raw = open(pgn_file, "r").read()
    lines = [line for line in raw.split("\n") if line.count("[") == 0]
    return "\n".join(lines)

processed = ""
for path in glob.glob("./fens/*.pgn"):
    processed += "    " + clean(path)

games = processed.split("\r\n\r\n")

fens = {}
for game_str in games:
    pgn = StringIO(u"%s" % game_str)
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
