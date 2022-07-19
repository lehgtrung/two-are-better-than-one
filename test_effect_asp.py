import subprocess
import json
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def split_data_for_ssl(in_path, out_path, portion):
    with open(in_path, 'r') as f:
        data = json.load(f)
    n = len(data)
    m = int(n*portion)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[:m]
    labeled = []
    unlabeled = []
    for i, row in enumerate(data):
        if i in indices:
            labeled.append(row)
        else:
            unlabeled.append(row)
    with open(out_path.format('labeled.json'), 'w') as f:
        json.dump(labeled, f)
    with open(out_path.format('unlabeled.json'), 'w') as f:
        json.dump(unlabeled, f)
    with open(out_path.format('indices.json'), 'w') as f:
        json.dump(indices, f)


def gen_data_folds(in_path):
    # Split the data into 30-70
    base_path = '../datasets/ssl_train_data/folds/{}'
    for i in tqdm(range(10)):
        path = base_path.format(i+1)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, '{}')
        split_data_for_ssl(in_path, out_path=path, portion=0.3)


def train_gen_labeled_data(index):
    # Constants
    raw_evaluation_path = f'../datasets/ssl_train_data/folds/{index}/raw_eval.json'
    verify_evaluation_path = f'../datasets/ssl_train_data/folds/{index}/verify_eval.json'

    base_model_path = f'../ckpts/folds/{index}/labeled_model'
    raw_model_path = f'../ckpts/folds/{index}/raw_model'
    verify_model_path = f'../ckpts/folds/{index}/verify_model'

    unlabeled_path = f'../datasets/ssl_train_data/folds/{index}/unlabeled.json'
    labeled_path = f'../datasets/ssl_train_data/folds/{index}/labeled.json'

    raw_inter_path = f'../datasets/ssl_train_data/folds/{index}/raw.json'
    verify_inter_path = f'../datasets/ssl_train_data/folds/{index}/verify.json'

    base_train_path = f'../datasets/ssl_train_data/folds/{index}/labeled.json'
    raw_train_path = f'../datasets/ssl_train_data/folds/{index}/raw_and_labeled.json'
    verify_train_path = f'../datasets/ssl_train_data/folds/{index}/verify_and_labeled.json'

    # Make dirs
    os.makedirs(f'../ckpts/folds/{index}', exist_ok=True)

    # Train a model from train_path, output raw/verified prediction to raw_pred_path/verified_path
    based_train_script = """
    python -u ../train.py \
    --num_layers 2 \
    --batch_size 2  \
    --evaluate_interval 1000 \
    --dataset CoNLL04 \
    --pretrained_wv ../wv/glove.6B.100d.conll04.txt \
    --max_epoches 2000 \
    --max_steps {} \
    --model_class JointModel \
    --model_write_ckpt {} \
    --crf None  \
    --optimizer adam \
    --lr 0.001  \
    --tag_form iob2 \
    --cased 0  \
    --token_emb_dim 100 \
    --char_emb_dim 30 \
    --char_encoder lstm  \
    --lm_emb_dim 4096 \
    --head_emb_dim 768 \
    --lm_emb_path ../wv/albert.conll04_with_heads.pkl \
    --hidden_dim 200     --ner_tag_vocab_size 9 \
    --re_tag_vocab_size 11     --vocab_size 15000     --dropout 0.5  \
    --grad_period 1 --warm_steps 1000 \
    --train_path {}
    """
    train_script = based_train_script.format(20000, base_model_path, base_train_path)
    print('Experiment #{}: Train on labeled data'.format(index))
    subprocess.run(train_script, shell=True, check=True)
    # Predict
    predict_script = f"""python ../predict_script.py {base_model_path} \
    {unlabeled_path} {raw_inter_path}
    """
    subprocess.run(predict_script, shell=True, check=True)
    # Verify
    verify_script = f"""python ../asp_script.py {raw_inter_path} \
    {unlabeled_path} {verify_inter_path} {raw_inter_path}
    """
    subprocess.run(verify_script, shell=True, check=True)

    # Union raw and labeled
    with open(labeled_path) as f:
        labeled = json.load(f)
    with open(raw_inter_path) as f:
        raw = json.load(f)
    with open(verify_inter_path) as f:
        verify = json.load(f)
    verify = verify + labeled
    raw = raw + labeled
    with open(verify_train_path, 'w') as f:
        json.dump(verify, f)
    with open(raw_train_path, 'w') as f:
        json.dump(raw, f)
    # Re-train on raw prediction
    print('Experiment #{}: Retrain on raw data'.format(index))
    raw_train_script = based_train_script.format(25000, raw_model_path, raw_train_path)
    subprocess.run(raw_train_script, shell=True, check=True)
    # Re-train on verified prediction
    print('Experiment #{}: Retrain on verified data'.format(index))
    verify_train_script = based_train_script.format(25000, verify_model_path, verify_train_path)
    subprocess.run(verify_train_script, shell=True, check=True)

    # Evaluation
    based_evaluation_script = """
        python -u ../evaluation.py \
        --num_layers 2 \
        --batch_size 2  \
        --evaluate_interval 1000 \
        --dataset CoNLL04 \
        --pretrained_wv ../wv/glove.6B.100d.conll04.txt \
        --max_epoches 5000 \
        --max_steps 1000 \
        --model_class JointModel \
        --model_read_ckpt {} \
        --crf None  \
        --optimizer adam \
        --lr 0.001  \
        --tag_form iob2 \
        --cased 0  \
        --token_emb_dim 100 \
        --char_emb_dim 30 \
        --char_encoder lstm  \
        --lm_emb_dim 4096 \
        --head_emb_dim 768 \
        --lm_emb_path ../wv/albert.conll04_with_heads.pkl \
        --hidden_dim 200     --ner_tag_vocab_size 9 \
        --re_tag_vocab_size 11     --vocab_size 15000     --dropout 0.5  \
        --grad_period 1 --warm_steps 0 > {}
        """
    raw_evaluation_script = based_evaluation_script.format(raw_model_path, raw_evaluation_path)
    verify_evaluation_script = based_evaluation_script.format(verify_model_path, verify_evaluation_path)

    subprocess.run(raw_evaluation_script, shell=True, check=True)
    subprocess.run(verify_evaluation_script, shell=True, check=True)


if __name__ == '__main__':
    # gen_data_folds('../datasets/unified/train.CoNLL04.json')
    train_gen_labeled_data(index=6)
    # train_gen_labeled_data(index=8)
    # train_gen_labeled_data(index=5)
    # for i in tqdm(range(10)):
    #     print('================================')
    #     print('Experiment #{}'.format(i+1))
    #     train_gen_labeled_data(index=i+1)


