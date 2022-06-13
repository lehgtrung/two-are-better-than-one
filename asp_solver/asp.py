
import subprocess
import json
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


clingo_path = '/home/thoang/anaconda3/bin/clingo'
clingo_options = ['--outf=2','-n 0']
clingo_command = [clingo_path] + clingo_options


def solve(program):
    input = program.encode()
    process = subprocess.Popen(clingo_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output, error = process.communicate(input)
    result = json.loads(output.decode())
    if result['Result'] == 'SATISFIABLE':
        return [value['Value'] for value in result['Call'][0]['Witnesses']]
    else:
        return None


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


def hash_entity(entity):
    etype = format_for_asp(entity[2], 'entity')
    eword = '_'.join(tokens[entity[0]:entity[1]])
    return '{}("{}").'.format(etype, eword)


def hash_relation(relation):
    rtype = format_for_asp(relation[4], 'relation')
    headword = '_'.join(tokens[relation[0]:relation[1]])
    tailword = '_'.join(tokens[relation[2]:relation[3]])
    return '{}("{}","{}").'.format(rtype, headword, tailword)


def create_inverted_index(entities, relations):
    emap = {}
    rmap = {}
    for entity in entities:
        e = hash_entity(entity)
        emap[e] = entity
    for relation in relations:
        r = hash_relation(relation)
        rmap[r] = relation
    return emap, rmap


def map_converted_back(mmap, obj, otype):
    ...


def convert_facts_to_asp_program(tokens, entities, relations, upto=5):
    # generate all words upto length 5
    words = []
    es = []
    rs = []
    for i in range(len(tokens)):
        for j in range(1, upto+1):
            if i+j < len(tokens):
                word = '_'.join(tokens[i:i+j])
                words.append(word)
    for entity in entities:
        e = hash_entity(entity)
        es.append(e)
    for relation in relations:
        r = hash_relation(relation)
        rs.append(r)
    return words, es, rs


def convert_subgraph_to_facts(sub_graph):
    wentities = []
    wrelations = []
    for node in sub_graph.nodes:
        e = '{}("{}").'.format(sub_graph.nodes[node]['etype'], node)
        wentities.append(e)
    for edge in sub_graph.edges:
        r = '{}("{}","{}").'.format(sub_graph.edges[edge]['rtype'], edge[0], edge[1])
        wrelations.append(r)
    return wentities, wrelations


def revise(entities, relations):
    # Convert facts to a graph and find connected components
    emap, rmap = create_inverted_index(entities, relations)
    final_entities = []
    final_relations = []
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
        sub_graph = graph.subgraph(compo)
        es, rs = convert_subgraph_to_facts(sub_graph)
        program = base_program + '\n' + concat_facts(es, rs)
        solutions = solve(program)
        if solutions:
            for e in es:
                final_entities.append(emap[e])
            for r in rs:
                final_relations.append(rmap[r])
    return final_entities, final_relations


def solution_stats(i, gt, pred, revised, solutions):
    gt = [e.strip('.') for e in gt]
    pred = [e.strip('.') for e in pred]
    revised = [e.strip('.') for e in revised]
    _solutions = set()
    for solution in solutions:
        for e in solution:
            _solutions.add(e)
    _solutions = list(_solutions)
    _solutions = [e.strip('.') for e in _solutions]

    inferred_facts = list(set(_solutions) - set(revised))
    accepted_facts = [e for e in pred if e in revised]
    rejected_facts = [e for e in pred if e not in revised]

    if len(rejected_facts) > 0 or len(inferred_facts) > 0 or len(set(pred) - set(gt)) > 0:
        print('Sentence: ', i)
        print('Ground truth: ', len(gt))
        print(gt)

        print('Predicted: ', len(pred))
        print(pred)

        print('Missing: ', len(list(set(gt) - set(pred))))
        print(list(set(gt) - set(pred)))

        print('Redundant: ', len(list(set(pred) - set(gt))))
        print(list(set(pred) - set(gt)))

        print('Accepted facts: ', len(accepted_facts))
        print(accepted_facts)

        print('Rejected facts: ', len(rejected_facts))
        print(rejected_facts)

        print('Inferred facts: ', len(inferred_facts))
        print(inferred_facts)

        print('Match ground truth: ', set(_solutions) == set(gt))

        print('Cover ground truth: ', set(_solutions).issuperset(set(gt)))
        print('========================================')

    return int(len(rejected_facts) > 0), \
           int(len(inferred_facts) > 0), \
           int(set(gt) == set(_solutions)), \
           int(set(_solutions).issuperset(set(gt))), \
           int((set(pred) - set(rejected_facts)) == set(gt)), \
           int((set(pred) - set(rejected_facts)).issuperset(set(gt)))


if __name__ == '__main__':
    with open('solver.lp') as f:
        base_program = f.read()

    with open('../datasets/ssl_outputs/predicted.CoNLL04_30_unlabeled.json') as f:
        pred_data = json.load(f)

    with open('../datasets/unified/train.CoNLL04_30_unlabeled.json') as f:
        gt_data = json.load(f)

    total_rejected = 0
    total_inferred = 0
    total_match = 0
    total_cover = 0
    total_pred_match = 0
    total_pred_cover = 0
    assert len(pred_data) == len(gt_data)
    for i, (pred_row, gt_row) in enumerate(zip(pred_data, gt_data)):
        tokens = gt_row['tokens']
        entities = pred_row['entity_preds']
        relations = pred_row['relation_preds']

        final_entities, final_relations = revise(entities, relations)
        words, es, rs = convert_facts_to_asp_program(tokens, final_entities, final_relations)

        program = base_program + '\n' + concat_facts(es, rs)
        solutions = solve(program)
        _, _es, _rs = convert_facts_to_asp_program(tokens, pred_row['entity_gts'], pred_row['relation_gts'])
        _gt = _es + _rs
        _, _es, _rs = convert_facts_to_asp_program(tokens, pred_row['entity_preds'], pred_row['relation_preds'])
        _pred = _es + _rs
        _revised = es + rs
        count_rejected, count_inferred, count_match, count_cover, count_pred_match, count_pred_cover = \
            solution_stats(i+1, _gt, _pred, _revised, solutions)
        total_rejected += count_rejected
        total_inferred += count_inferred
        total_match += count_match
        total_cover += count_cover
        total_pred_match += count_pred_match
        total_pred_cover += count_pred_cover

    print('Total sentences with rejected facts:', total_rejected)
    print('Percentage sentences with rejected facts:', total_rejected / len(gt_data) * 100)
    print()
    print('Total sentences with inferred facts:', total_inferred)
    print('Percentage sentences with inferred facts:', total_inferred / len(gt_data) * 100)
    print()
    print('Total inferred sentences match ground truth:', total_match)
    print('Percentage sentences match ground truth:', total_match / len(gt_data) * 100)
    print()
    print('Total inferred sentences cover ground truth:', total_cover)
    print('Percentage sentences cover ground truth:', total_cover / len(gt_data) * 100)
    # print()
    # print('Total predicted sentences match ground truth:', total_pred_match)
    # print('Percentage predicted sentences match ground truth:', total_pred_match / len(gt_data) * 100)
    # print()
    # print('Total predicted sentences cover ground truth:', total_pred_cover)
    # print('Percentage predicted sentences cover ground truth:', total_pred_cover / len(gt_data) * 100)





