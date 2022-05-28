import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


def extend_last_dim(tensor, value):
    _shape = list(tensor.shape)
    _shape[-1] = 1
    _tensor = torch.ones(_shape)
    _tensor = _tensor * (value)
    _tensor = torch.cat((tensor, _tensor), dim=-1)
    return _tensor


def invert_index_with_undecided(tags, preds):
    max_values = max(tags.values())
    tags['U'] = max_values + 1
    tags = dict([(value, key) for key, value in tags.items()])
    return np.vectorize(tags.get)(preds)


def make_prediction_with_undecided(logits, mask, tags, threshold):
    epsilon = 1e-3
    logits = torch.softmax(logits, dim=-1)
    logits = extend_last_dim(logits, threshold - epsilon)
    preds = logits.argmax(dim=-1).cpu().detach().numpy()
    preds *= mask
    preds = invert_index_with_undecided(tags, preds)
    return preds