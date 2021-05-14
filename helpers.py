import chess
import numpy as np

def count_moves(row):
    """Input: Dataframe row. Output: Integer"""
    moves = row['Moves'].split(' ')
    move_count = len(moves)
    return move_count

def make_array(fen: str, piece_dict: dict) -> np.array:
    """Create a numpy array from the FEN."""
    board = chess.Board(fen)
    x = str(board)
    x = x.replace('\n', '')
    x = x.replace(' ', '')
    arr = np.array([piece_dict[c] for c in x])
    return arr

def col_to_set(col: str) -> set:
    """Input: string dataframe column. Output: set of the unique items."""
    _set = set(col.split(' '))
    return _set

def clean_labels(input_col: str, lbls_to_remove: list) -> str:
    """Input: string of themes, themes to remove. Output: cleaned themes."""
    input_col = input_col.lower()
    for lbl in lbls_to_remove:
        if lbl in input_col:
            input_col = input_col.replace(lbl, '')
            input_col = input_col.replace('  ', ' ').strip()
    return input_col

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(1024, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam') # binary_crossentropy
    return model
