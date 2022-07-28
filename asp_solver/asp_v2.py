import subprocess
import json
import ast
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
from asp_ult import *


def convert_solution_to_data(tokens, solution):
    data_point = {
        'tokens': tokens,
        'entities': [],
        'relations': []
    }
    for atom in solution:
        if match_form(atom) == 'entity':
            entity_type, word = extract_from_atom(atom, 'entity')
            start, end = word.split('+')
            data_point['entities'].append([
                start,
                end,
                polish_type(entity_type)
            ])
        else:
            relation_type, head_word, tail_word = extract_from_atom(atom, 'relation')
            hstart, hend = head_word.split('+')
            tstart, tend = tail_word.split('+')
            data_point['relations'].append([
                hstart,
                hend,
                tstart,
                tend,
                polish_type(relation_type)
            ])
    return data_point


def convert_solutions_back(solution):
    es = []
    rs = []
    for atom in solution:
        atom = atom.replace('ok(', '', 1).replace(')', '', 1) + '.'
        if atom.startswith('loc(') or atom.startswith('peop(') or \
                atom.startswith('org(') or atom.startswith('other('):
            es.append(atom)
        else:
            rs.append(atom)
    return es, rs


def verify_and_infer(entities, relations, inference_program):
    final_outputs = []

    # Remove connected components
    es = convert_original_to_atoms(entities, 'entity')
    rs = convert_original_to_atoms(relations, 'relation')
    program = concat_facts(es, rs)
    answer_sets = solve_v2(program)
    for answer_set in answer_sets:
        es, rs = convert_solutions_back(answer_set)
        program = inference_program + '\n' + concat_facts(es, rs)
        solution = solve(program)
        if not solution:
            continue
        solution = ['ok(' + atom + ')' for atom in solution]
        es, rs = convert_solutions_back(solution)
        final_outputs.append(es + rs)
    return final_outputs


def compute_atom_weight(answer_sets, uniform=False):
    # Number of times an atom appears in each answer_set / total number of answer sets
    united_atoms = []
    weights = []
    if uniform:
        n = len(answer_sets)
        if n == 0:
            return [], []
        print('Number of answer sets: ', n)
        concat_answer_sets = np.concatenate(answer_sets).tolist()
        unique_concat_answer_sets = list(set(concat_answer_sets))
        for atom in unique_concat_answer_sets:
            if concat_answer_sets.count(atom) == n:
                united_atoms.append(atom)
        weights = [1.0 for _ in range(len(united_atoms))]
        return united_atoms, weights
    else:
        for answer_set in answer_sets:
            for atom in answer_set:
                united_atoms.append(atom)
        for atom in united_atoms:
            weight = 0
            for answer_set in answer_sets:
                if atom in answer_set:
                    weight += 1
            weights.append(weight / len(answer_sets))
        return united_atoms, weights


if __name__ == '__main__':
    with open('exp_area/p_star.lp') as f:
        verification_program = f.read()

    with open('inference.lp') as f:
        inference_program = f.read()

    with open('../datasets/ssl_outputs/argmax_predicted.CoNLL04_30_unlabeled.json') as f:
        pred_data = json.load(f)

    with open('../datasets/unified/train.CoNLL04_30_unlabeled.json') as f:
        gt_data = json.load(f)

    assert len(pred_data) == len(gt_data)
    print('Length: ', len(gt_data))
    count_s_equal_t = 0
    count_false_true = 0
    count_p_equal_t = 0
    pred_iou = []
    solution_iou = []
    data_points = []
    for i, (pred_row, gt_row) in enumerate(zip(pred_data, gt_data)):
        print('=============================')
        print(i)

        tokens = gt_row['tokens']
        entities = pred_row['entity_preds']
        relations = pred_row['relation_preds']

        print(convert_original_to_atoms(entities, 'entity'))
        print(convert_original_to_atoms(relations, 'relation'))

        final_outputs = verify_and_infer(entities, relations, inference_program)

        united_atoms, atom_weights = compute_atom_weight(final_outputs, uniform=True)

        print(final_outputs)
        print(united_atoms)
        print(atom_weights)

        data_point = convert_solution_to_data(tokens, united_atoms)

        # Convert solution to new data
        data_point = {
            'tokens': data_point['tokens'],
            'entities': data_point['entities'],
            'relations': data_point['relations'],
            'id': i
        }
        data_points.append(data_point)

    # with open('../datasets/ssl_train_data/argmax_w_all_answersets_with_intersection_complete.json', 'w') as f:
    #     json.dump(data_points, f)
