import subprocess
import json
import ast
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools

clingo_path = '/home/thoang/anaconda3/bin/clingo'
clingo_options = ['--outf=2', '-n 0']
clingo_command = [clingo_path] + clingo_options

drive_command = ['/home/thoang/anaconda3/bin/clingo', 'exp_area/drive.py', 'exp_area/p_star.lp', '--outf=3']


def union_all_solutions(solutions):
    union = set()
    for solution in solutions:
        for atom in solution:
            if not atom.startswith('nOfOKAtoms'):
                union.add(atom)
    return list(union)


def solve(program):
    input = program.encode()
    process = subprocess.Popen(clingo_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output, error = process.communicate(input)
    result = json.loads(output.decode())
    if result['Result'] == 'SATISFIABLE' or result['Result'] == 'OPTIMUM FOUND':
        solutions = [value['Value'] for value in result['Call'][0]['Witnesses']]
        return union_all_solutions(solutions)
    else:
        return None


def solve_v2(program):
    # Write the program to a file
    with open('exp_area/p_star.lp', 'w') as f:
        f.write(program)
    process = subprocess.Popen(drive_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output, error = process.communicate()
    result = ast.literal_eval(output.decode().split('\n')[-2])
    return result


def format_for_asp(s, type):
    if type == 'entity':
        return s.lower()
    else:
        splits = s.split('_')
        if len(splits) > 1:
            return '{}{}'.format(splits[0].lower(), splits[1].capitalize())
        return splits[0].lower()


def concat_facts(es, rs):
    output = []
    for e in es:
        output.append(e)
    for r in rs:
        output.append(r)
    return '\n'.join(output)


def hash_entity(entity, with_atom):
    etype = format_for_asp(entity[2], 'entity')
    eword = '_'.join(tokens[entity[0]:entity[1]])
    if with_atom:
        return 'atom({}("{}")).'.format(etype, eword)
    return '{}("{}").'.format(etype, eword)


def hash_relation(relation, with_atom):
    rtype = format_for_asp(relation[4], 'relation')
    headword = '_'.join(tokens[relation[0]:relation[1]])
    tailword = '_'.join(tokens[relation[2]:relation[3]])
    if with_atom:
        return 'atom({}("{}","{}")).'.format(rtype, headword, tailword)
    return '{}("{}","{}").'.format(rtype, headword, tailword)


def create_inverted_index(entities, relations):
    emap = {}
    rmap = {}
    for entity in entities:
        e = hash_entity(entity, with_atom=False)
        emap[e] = entity
    for relation in relations:
        r = hash_relation(relation, with_atom=False)
        rmap[r] = relation
    return emap, rmap


def match_form(atom):
    open_pos = atom.index('(')
    if atom[:open_pos] in ['peop', 'loc', 'org', 'other']:
        return 'entity'
    return 'relation'


def polish_type(atom_type):
    if atom_type in ['peop', 'loc', 'org', 'other']:
        return atom_type.capitalize()
    if atom_type == 'liveIn':
        return 'Live_In'
    elif atom_type == 'locatedIn':
        return 'Located_In'
    elif atom_type == 'orgbasedIn':
        return 'OrgBased_In'
    elif atom_type == 'workFor':
        return 'Work_For'
    return 'Kill'


def extract_from_atom(atom, form_type):
    open_pos = atom.index('(')
    close_pos = atom.index(')')
    if form_type == 'entity':
        return atom[:open_pos], atom[open_pos + 1:close_pos].strip().strip('"')
    # count number of comma
    count = atom.count(',')
    if count == 1:
        comma_pos = atom.index(',')
    else:
        comma_pos = atom.index('",') + 1
    return atom[:open_pos], \
           atom[open_pos + 1:comma_pos].strip().strip('"'), \
           atom[comma_pos + 1:close_pos].strip().strip('"')


def find_word_position(tokens, word):
    word = word.split('_')
    n = len(tokens)
    m = len(word)
    for i in range(n - m + 1):
        if tokens[i] == word[0]:
            if tokens[i + 1:i + m] == word[1:]:
                return i, i + m


def convert_solution_to_data(tokens, solution):
    data_point = {
        'tokens': tokens,
        'entities': [],
        'relations': []
    }
    for atom in solution:
        if match_form(atom) == 'entity':
            entity_type, word = extract_from_atom(atom, 'entity')
            start, end = find_word_position(tokens, word)
            data_point['entities'].append([
                start,
                end,
                polish_type(entity_type)
            ])
        else:
            relation_type, head_word, tail_word = extract_from_atom(atom, 'relation')
            hstart, hend = find_word_position(tokens, head_word)
            tstart, tend = find_word_position(tokens, tail_word)
            data_point['relations'].append([
                hstart,
                hend,
                tstart,
                tend,
                polish_type(relation_type)
            ])
    return data_point


def convert_facts_to_asp_program(tokens, entities, relations, upto=5):
    # generate all words upto length 5
    words = []
    es = []
    rs = []
    for i in range(len(tokens)):
        for j in range(1, upto + 1):
            if i + j < len(tokens):
                word = '_'.join(tokens[i:i + j])
                words.append(word)
    for entity in entities:
        e = hash_entity(entity, with_atom=False)
        es.append(e)
    for relation in relations:
        r = hash_relation(relation, with_atom=False)
        rs.append(r)
    return words, es, rs


def convert_subgraph_to_facts(sub_graph):
    wentities = []
    wrelations = []
    for node in sub_graph.nodes:
        e = 'atom({}("{}")).'.format(sub_graph.nodes[node]['etype'], node)
        wentities.append(e)
    for edge in sub_graph.edges:
        r = 'atom({}("{}","{}")).'.format(sub_graph.edges[edge]['rtype'], edge[0], edge[1])
        wrelations.append(r)
    return wentities, wrelations


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


def verify_and_infer(entities, relations, verification_program, inference_program):
    # Convert facts to a graph and find connected components
    emap, rmap = create_inverted_index(entities, relations)
    final_outputs = []
    graph = nx.DiGraph()
    for entity in entities:
        etype = format_for_asp(entity[2], 'entity')
        eword = '_'.join(tokens[entity[0]:entity[1]])
        graph.add_node(eword, etype=etype)
    for relation in relations:
        rtype = format_for_asp(relation[4], 'relation')
        headword = '_'.join(tokens[relation[0]:relation[1]])
        tailword = '_'.join(tokens[relation[2]:relation[3]])
        graph.add_edge(headword, tailword, rtype=rtype)

    components = nx.weakly_connected_components(graph)
    for compo in components:
        comp_output = []
        sub_graph = graph.subgraph(compo)
        es, rs = convert_subgraph_to_facts(sub_graph)
        program = verification_program + '\n' + concat_facts(es, rs)
        answer_sets = solve_v2(program)
        for answer_set in answer_sets:
            es, rs = convert_solutions_back(answer_set)
            program = inference_program + '\n' + concat_facts(es, rs)
            solution = solve(program)
            solution = ['ok(' + atom + ')' for atom in solution]
            es, rs = convert_solutions_back(solution)
            #print(es)
            #print(rs)
            #es = [emap[e] for e in es]
            #rs = [rmap[r] for r in rs]
            comp_output.append(es + rs)
        final_outputs.append(comp_output)
    final_outputs = answer_set_product(final_outputs)
    return final_outputs


def set_iou(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    i = len(set1.intersection(set2))
    u = len(set1.union(set2))
    return i / u


def answer_set_product(compos):
    result = []
    compos = sorted(compos, key=len, reverse=True)
    rep = [tuple(range(len(compo))) for compo in compos]
    products = list(itertools.product(*rep))
    for product in products:
        _temp = []
        for j, i in enumerate(product):
            _temp += compos[j][i]
        result.append(_temp)
    return result


def compute_atom_weight(answer_sets):
    # Number of times an atom appears in each answer_set / total number of answer sets
    united_atoms = []
    weights = []
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


def split_entities_relations(atoms, weights):
    entities = []
    entity_weights = []
    relations = []
    relation_weights = []
    for atom, weight in zip(atoms, weights):
        if match_form(atom) == 'entity':
            entities.append(atom)
            entity_weights.append(weight)
        else:
            relations.append(atom)
            relation_weights.append(weight)
    return entities, entity_weights, relations, relation_weights


if __name__ == '__main__':
    with open('verification.lp') as f:
        verification_program = f.read()

    with open('inference.lp') as f:
        inference_program = f.read()

    with open('../datasets/ssl_outputs/argmax_predicted.CoNLL04_30_unlabeled.json') as f:
        pred_data = json.load(f)

    with open('../datasets/unified/train.CoNLL04_30_unlabeled.json') as f:
        gt_data = json.load(f)

    # raw_pred_data = []
    # for i, (pred_row, gt_row) in enumerate(zip(pred_data, gt_data)):
    #     tokens = gt_row['tokens']
    #     entities = pred_row['entity_preds']
    #     relations = pred_row['relation_preds']
    #
    #     _, _entities1, _relations1 = convert_facts_to_asp_program(tokens, entities, relations)
    #     pred_set = sorted(_entities1 + _relations1)
    #     pred_set = convert_solution_to_data(tokens, pred_set)
    #     raw_pred_data.append(pred_set)
    #
    # with open('../datasets/ssl_outputs/argmax_wo_ssl.json', 'w') as f:
    #     json.dump(raw_pred_data, f)
    #
    # exit()

    assert len(pred_data) == len(gt_data)
    print('Length: ', len(gt_data))
    count_s_equal_t = 0
    count_false_true = 0
    count_p_equal_t = 0
    pred_iou = []
    solution_iou = []
    data_points = []
    for i, (pred_row, gt_row) in enumerate(zip(pred_data, gt_data)):
        if i in [227, 229, 276, 277, 501]:
            continue
        print(i)
        # print('=============================')
        tokens = gt_row['tokens']
        entities = pred_row['entity_preds']
        relations = pred_row['relation_preds']

        # print('tokens: ', tokens)
        # print('entities: ', entities)
        # print('relations: ', relations)
        # continue

        final_outputs = verify_and_infer(entities, relations,
                                         verification_program, inference_program)
        united_atoms, atom_weights = compute_atom_weight(final_outputs)

        data_point = convert_solution_to_data(tokens, united_atoms)
        _, final_eweights, _, final_rweights = split_entities_relations(united_atoms, atom_weights)

        assert len(final_eweights) == len(data_point['entities'])
        assert len(final_rweights) == len(data_point['relations'])

        # Convert solution to new data
        data_point = {
            'tokens': data_point['tokens'],
            'entities': data_point['entities'],
            'relations': data_point['relations'],
            'eweights': final_eweights,
            'rweights': final_rweights
        }
        data_points.append(data_point)

    with open('../datasets/ssl_train_data/argmax_w_all_answersets_with_inference.json', 'w') as f:
        json.dump(data_points, f)
