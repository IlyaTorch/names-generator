import torch
import os

from random import sample
from model.CharRNNCell import rnn_loop, CharRNNCell
from model.words_preparing import lines_to_matrix, read_names, get_unique_tokens, get_token_id_dict


def save_model(model):
    name = str(model.__class__)
    name = name[name.find('.')+1:name.find(">")-1]

    path = f"model/resources/checkpoint_{name}.pth"

    model_state = {
                    'model_name': name,
                    'net': model.state_dict(),
                }
    if not os.path.isdir('../resources/'):
        os.mkdir('../resources/')
    torch.save(model_state, path)


def train(lines, token_to_id, model, optimizer, num_epoch=1000):
    model.train()
    history = []

    for i in range(num_epoch):
        batch_ix = lines_to_matrix(sample(lines, 32), token_to_id)
        batch_ix = torch.tensor(batch_ix, dtype=torch.int64)

        logp_seq = rnn_loop(model, batch_ix)

        # compute loss
        predictions_logp = logp_seq[:, :-1]
        actual_next_tokens = batch_ix[:, 1:]
        logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
        loss = -logp_next.mean()

        # train with backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.data.numpy())

    return history


if __name__ == '__main__':
    lines = read_names('model/resources/names.txt')
    tokens = get_unique_tokens(lines)
    token_to_id = get_token_id_dict(tokens)

    model = CharRNNCell(len(tokens))
    optimizer = torch.optim.Adam(model.parameters())
    train(lines, token_to_id, model, optimizer, 5000)
    save_model(model)
