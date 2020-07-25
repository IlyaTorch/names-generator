import numpy as np


def get_unique_tokens(lines: list):
    return list(np.unique([c for word in lines for c in word]))


def get_token_id_dict(tokens: list):
    return {str(token): tokens.index(token) for token in tokens}


def read_names(path: str, start_token=" "):
    with open(path) as f:
        lines = f.read()[:-1].split('\n')
        lines = [start_token + line for line in lines]
        return lines


def lines_to_matrix(lines: list, token_to_id:dict, max_len=None, dtype='int32'):
    """Casts a list of names into rnn-digestable matrix"""
    pad = token_to_id[' ']
    max_len = max_len or max(map(len, lines))  # all the words must have the same length

    matrix_indexes = np.zeros([len(lines), max_len], dtype)
    matrix_indexes += pad  # initialising by indexes of padding

    for i, word in enumerate(lines):
        line_index = [token_to_id[c] for c in word]  # word to list of numbers
        matrix_indexes[i, :len(line_index)] = line_index

    return matrix_indexes
