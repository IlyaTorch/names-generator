import numpy as np
import torch
import torch.nn.functional as F

from CharRNNCell import CharRNNCell
from words_preparing import read_names, get_unique_tokens, get_token_id_dict


def generate_sample(char_rnn, tokens, token_to_id, max_length, seed_phrase=' ',
                    temperature=1.0):  # temperature - parameter for the softmax
    num_tokens = len(tokens)
    x_sequence = [token_to_id[token] for token in seed_phrase]
    x_sequence = torch.tensor([x_sequence], dtype=torch.int64)
    hid_state = char_rnn.initial_state(batch_size=1)

    # feed the seed phrase, if any
    for i in range(len(seed_phrase) - 1):
        hid_state, _ = char_rnn(x_sequence[:, i], hid_state)

    # start generating
    for _ in range(max_length - len(seed_phrase)):
        hid_state, logp_next = char_rnn(x_sequence[:, -1], hid_state)
        p_next = F.softmax(logp_next / temperature, dim=-1).data.numpy()[0]

        # sample next token and push it back into x_sequence
        next_ix = np.random.choice(num_tokens, p=p_next)  # из массива номеров букв выбираем по вероятности символ
        next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
        x_sequence = torch.cat([x_sequence, next_ix], dim=1)

    return ''.join([tokens[ix] for ix in x_sequence.data.numpy()[0]])


def inference(seed_phrase=' ', num_examples=5, path='resources/names.txt'):
    lines = read_names(path)
    tokens = get_unique_tokens(lines)
    token_to_id = get_token_id_dict(tokens)
    max_length = max(map(len, lines))

    char_rnn = CharRNNCell(len(tokens))
    checkpoint = torch.load("resources/checkpoint_CharRNNCell.pth")
    char_rnn.load_state_dict(checkpoint['net'])

    samples = [generate_sample(char_rnn, tokens, token_to_id, max_length, seed_phrase) for _ in range(num_examples)]
    samples = [name.strip() for name in samples]
    return samples


if __name__ == '__main__':
    print(inference(' J', 10))
