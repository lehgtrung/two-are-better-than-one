import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from utils import *
from data import *
from models import *

# torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)


def make_prediction(model, input_path, output_path):
    kept_fields = ['entity_preds', 'relation_preds']
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    outputs = []
    for row in tqdm(input_data):
        tokens = row['tokens']
        step_input = {
            'tokens': [tokens]
        }
        step_output = {
            'entity_preds': [],
            'relation_preds': [],
            'entity_gts': [],
            'relation_gts': []
        }
        rets = model.predict_step(step_input)
        rets = {k: list(v[0]) for k, v in rets.items() if k in kept_fields}
        # Append the gt
        for e in row['entities']:
            k = list(e)
            k.append(' '.join(tokens[e[0]: e[1]]))
            step_output['entity_gts'].append(k)
        for r in row['relations']:
            k = list(r)
            k.append(' '.join(tokens[r[0]: r[1]]))
            k.append(' '.join(tokens[r[2]: r[3]]))
            step_output['relation_gts'].append(k)

        # Append the predicted
        for e in rets['entity_preds']:
            k = list(e)
            k.append(' '.join(tokens[e[0]: e[1]]))
            step_output['entity_preds'].append(k)
        for r in rets['relation_preds']:
            k = list(r)
            k.append(' '.join(tokens[r[0]: r[1]]))
            k.append(' '.join(tokens[r[2]: r[3]]))
            step_output['relation_preds'].append(k)
        outputs.append(step_output)
    with open(output_path, 'w') as f:
        json.dump(outputs, f)


# ********* MODIFY HERE *********
args = {
    'model_read_ckpt': './ckpts/conll04_30',
    'lm_emb_path': 'albert-xxlarge-v1', # language model name
    'pretrained_wv': './wv/glove.6B.100d.conll04.txt', # original GloVe embeddings
    'vocab_size': 400100, # GloVe contains 400,000 words
    'device': 'cpu',
}

if args['device'] is not None and args['device'] != 'cpu':
    torch.cuda.set_device(args['device'])
elif args['device'] is None:
    if torch.cuda.is_available():
        gpu_idx, gpu_mem = set_max_available_gpu()
        args['device'] = f"cuda:{gpu_idx}"
    else:
        args['device'] = "cpu"

# Load the config file of trained ckpt
with open(args['model_read_ckpt'] + '.json', 'r') as f:
    config = Config(**json.load(f))

    # load language model to dynamically calculate the contextualized word embeddings
    config.lm_emb_path = args['lm_emb_path']
    # assign device
    config.device = args['device']

model = JointModel(config)
model.load_ckpt(args['model_read_ckpt'])


# Load full GloVe embeddings
# *this is needed when training on the reduced version of GloVe word vectors
# *you can comment this block if the OOV problem is not very serious.
_w = model.token_embedding.token_embedding.weight
_w_data = _w.data
_w.data = torch.zeros([args['vocab_size'], config.token_emb_dim], dtype=_w.dtype, device=_w.device)
model.token_embedding.load_pretrained(args['pretrained_wv'], freeze=True)
_w.data[:len(_w_data)] = _w_data

# Now predict on custom text
# rets = model.predict_step({
#     'tokens': [["Newspaper", "`", "Explains", "'", "U.S.", "Interests", "Section", "Events", "FL1402001894",
#                 "Havana", "Radio", "Reloj", "Network", "in", "Spanish", "2100", "GMT", "13", "Feb", "94"],]
# })
#
# print(rets['entity_preds'])
# print(rets['relation_preds'])
# print(rets['ner_tag_preds'])
# print(rets['re_tag_preds'])

if __name__ == '__main__':
    make_prediction(model,
                    'datasets/unified/train.CoNLL04_30_unlabeled.json',
                    'datasets/ssl_outputs/argmax_predicted.CoNLL04_30_unlabeled.json')
