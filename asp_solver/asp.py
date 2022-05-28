
import subprocess
import json
from tqdm import tqdm
import networkx as nx


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
        etype = format_for_asp(entity[2], 'entity')
        eword = '_'.join(tokens[entity[0]:entity[1]])
        e = '{}("{}").'.format(etype, eword)
        es.append(e)
    for relation in relations:
        rtype = format_for_asp(relation[4], 'relation')
        headword = '_'.join(tokens[relation[0]:relation[1]])
        tailword = '_'.join(tokens[relation[2]:relation[3]])
        r = '{}("{}","{}").'.format(rtype, headword, tailword)
        rs.append(r)
    return words, es, rs


def convert_subgraph_to_facts(sub_graph):
    wentities = []
    wrelations = []
    for node in sub_graph.nodes:
        e = '{}("{}").'.format(node, sub_graph.nodes[node]['etype'])
        wentities.append(e)
    for edge in sub_graph.edges:
        r = '{}("{}","{}").'.format(sub_graph.edges[edge]['rtype'], edge[0], edge[1])
        wrelations.append(r)
    return wentities, wrelations


def revise(entities, relations):
    # Convert facts to a graph and find connected components
    final_entities = []
    final_relations = []
    graph = nx.Graph()
    for entity in entities:
        etype = format_for_asp(entity[2], 'entity')
        eword = '_'.join(tokens[entity[0]:entity[1]])
        graph.add_node(eword, etype=etype)
    for relation in relations:
        rtype = format_for_asp(relation[4], 'relation')
        headword = '_'.join(tokens[relation[0]:relation[1]])
        tailword = '_'.join(tokens[relation[2]:relation[3]])
        graph.add_edge(headword, tailword, rtype=rtype)
    components = nx.connected_components(graph)
    for compo in components:
        sub_graph = graph.subgraph(compo)
        es, rs = convert_subgraph_to_facts(sub_graph)
        program = base_program + '\n' + concat_facts(es, rs)
        solutions = solve(program)
        if solutions:
            final_entities.append(es)
            final_relations.append(rs)
    return final_entities, final_relations


if __name__ == '__main__':
    with open('solver.lp') as f:
        base_program = f.read()

    with open('../datasets/ssl_outputs/predicted.CoNLL04_30_unlabeled.json') as f:
        pred_data = json.load(f)

    with open('../datasets/unified/train.CoNLL04_30_unlabeled.json') as f:
        gt_data = json.load(f)

    total_unsatisfiable = 0
    total_inferred = 0
    assert len(pred_data) == len(gt_data)
    for pred_row, gt_row in zip(pred_data, gt_data):
        # print('========================')
        tokens = gt_row['tokens']
        entities = pred_row['entity_preds']
        relations = pred_row['relation_preds']
        words, es, rs = convert_facts_to_asp_program(tokens, entities, relations)

        # print(concat_facts(es, rs))
        program = base_program + '\n' + concat_facts(es, rs)
        solutions = solve(program)
        if solutions:
            if len(es) + len(rs) < len(solutions):
                total_inferred += 1
                print(concat_facts(es, rs))
                for i, solution in enumerate(solutions):
                    print(i + 1, solution)
                print('========================')
        else:
            print('Program UNSATISFIABLE')
            print(concat_facts(es, rs))
            print('========================')
            total_unsatisfiable += 1
        # exit()
    print('Total UNSATISFIABLE:', total_unsatisfiable)
    print('Percentage UNSATISFIABLE:', total_unsatisfiable / len(gt_data))
    print('Total INFERRED:', total_inferred)
    print('Percentage INFERRED:', total_inferred / len(gt_data))





