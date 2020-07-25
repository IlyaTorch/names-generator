import torch
import torch.nn as nn
import torch.nn.functional as F


class CharRNNCell(nn.Module):
    """Representation of the recurrent neural network's cell"""

    def __init__(self, num_tokens, embedding_size=16, rnn_num_units=64):
        super(self.__class__, self).__init__()
        self.num_units = rnn_num_units

        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.rnn_update = nn.Linear(embedding_size + rnn_num_units, rnn_num_units)
        self.rnn_to_logits = nn.Linear(rnn_num_units, num_tokens)

    def forward(self, x, h_prev):
        """
        h_next(x, h_prev) and log P(x_next | h_next)
        :param x: batch of character indexes, int64[batch_size]
        :param h_prev: previous rnn hidden states, float32 matrix [batch, rnn_num_units]
        """

        x_emb = self.embedding(x)
        h_next = self.rnn_update(torch.cat((x_emb, h_prev), dim=1))
        h_next = torch.tanh(h_next)

        logits = self.rnn_to_logits(h_next)

        return h_next, F.log_softmax(logits, -1)

    def initial_state(self, batch_size):
        """ :returns rnn state before it processes first input (h0) """
        return torch.zeros(batch_size, self.num_units)


def rnn_loop(char_rnn, batch_ix):
    batch_size, max_length = batch_ix.size()
    hid_state = char_rnn.initial_state(batch_size)
    log_probabilities = []

    for x_t in batch_ix.transpose(0, 1):  # for parallel prediction of the next char for each word
        hid_state, logp_next = char_rnn(x_t, hid_state)  # one_step in the loop
        log_probabilities.append(logp_next)

    return torch.stack(log_probabilities, dim=1)
