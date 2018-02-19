# ChessClassifier

Based on SVHNClassifier, a TensorFlow implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/pdf/1312.6082.pdf). Here, instead of recognizing house numbers in Street View images, we're recognizing pieces in chessboards. So for instance the goal is to take an image like this:

![image](https://user-images.githubusercontent.com/21294/36359922-909faea8-14ed-11e8-8edc-767f408353af.png)

and figure out what all the pieces are. Here's one representation, where "X" means "blank square," capital letters are white pieces, and lowercase letters are black pieces:

```
XXXXqXXX
nXrXnXXk
XppXprpp
pXXXXpXX
XXBPXNXX
PXXXPPXX
XPXXXPRP
XQXXXXRK
```

More compactly, but less readably, we could describe the board as `4q3-n1r1n2k-1pp1prpp-p4p2-2BP1N2-P3PP2-1P3PRP-1Q4RK` in [the classic notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation).

So the goal of our neural network is to take in images of boards and spit out one of those strings.

This turns out to be closely analogous to the Street View House Number problem: we have an image with "digits" (chess pieces) in it, we don't know where; we want to both find and identify those "digits."

Whereas the SVHN classifier had 6 softmaxes as its output, one of size 5 to say how many digits the house number was (1 through 5) and five softmaxes of size 10, one for each of the possible digits... we have 64 softmaxes of size 13 as output, one for each square on the board. (The 13 comes from the 6 possible chess pieces -- pawn, rook, knight, bishop, queen, king -- times 2 for each of the colors (black or white), plus the empty square.)

## Graph

Whereas the SVHNClassifier's graph looked like this:

![Graph](https://github.com/potterhsu/SVHNClassifier/blob/master/images/graph.png?raw=true)

Ours is:

![image](https://user-images.githubusercontent.com/21294/36360126-20347ae8-14ef-11e8-9bbb-08abe8bb93a6.png)

which if you zoomed out you'd see 64 separate "piece" outputs at the top.

## Results

### Accuracy

![image](https://user-images.githubusercontent.com/21294/36360149-4fb5d5a0-14ef-11e8-896c-e942162d98ce.png)

> Accuracy 99+% on test dataset after about 20 hours

### Loss

After some false starts, the loss started dropping:

![image](https://user-images.githubusercontent.com/21294/36360162-66dc38be-14ef-11e8-828e-ea4e73b79d65.png)

## Usage

See https://github.com/potterhsu/SVHNClassifier for detailed usage instructions. The difference here is that the data is generated using the *generate_data.py* script from PGNs (chess game files) downloaded off the internet from this page: http://www.pgnmentor.com/files.html.
