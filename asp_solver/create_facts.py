
import json


if __name__ == '__main__':
    with open('../datasets/ssl_outputs/predicted.CoNLL04_30_unlabeled.json') as f:
        pred_data = json.load(f)

    with open('../datasets/unified/train.CoNLL04_30_unlabeled.json') as f:
        gt_data = json.load(f)

    for i, (pred_row, gt_row) in enumerate(zip(pred_data, gt_data)):
        tokens = gt_row['tokens']
        entities = pred_row['entity_preds']
        relations = pred_row['relation_preds']

