
import json
from asp_ult import convert_original_to_atoms, format_for_asp


def compute_iou():
    pass


def check_intersection_iou():
    ...


def check_standalone_entities(data):
    # Check if there is any standalone non-other entity
    for i, line in enumerate(data):
        for rel in data['relations']:
            ...


def standardize_dataset(data):
    for i, line in enumerate(data):
        line['id'] = i
        ent_atoms = convert_original_to_atoms(line['entity_preds'], 'entity')
        rel_atoms = convert_original_to_atoms(line['relation_preds'], 'relation')
        s = ''.join(ent_atoms + rel_atoms)
        line['atoms'] = s
    return data


if __name__ == '__main__':
    with open('../datasets/ssl_outputs/argmax_predicted.CoNLL04_30_unlabeled.bk.json', 'r') as f:
        data = json.load(f)
        data = standardize_dataset(data)
    with open('../datasets/ssl_outputs/argmax_predicted.CoNLL04_30_unlabeled.json', 'w') as f:
        json.dump(data, f)