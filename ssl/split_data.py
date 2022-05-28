
import json
import random


def split_data(in_path, out_path, portion):
    with open(in_path, 'r') as f:
        data = json.load(f)
    splitted_data = random.sample(data, k=int(len(data) * portion))
    with open(out_path, 'w') as f:
        json.dump(splitted_data, f)


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
    with open(out_path.format('labeled'), 'w') as f:
        json.dump(labeled, f)
    with open(out_path.format('unlabeled'), 'w') as f:
        json.dump(unlabeled, f)


if __name__ == '__main__':
    split_data_for_ssl('../datasets/unified/train.CoNLL04.json',
                       '../datasets/unified/train.CoNLL04_30_{}.json', 0.3)




