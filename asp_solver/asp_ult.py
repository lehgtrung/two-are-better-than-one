import json


def format_for_asp(s, type):
    if type == 'entity':
        return s.lower()
    else:
        splits = s.split('_')
        if len(splits) > 1:
            return '{}{}'.format(splits[0].lower(), splits[1].capitalize())
        return splits[0].lower()


def convert_original_to_atoms(data, dtype):
    result = []
    for d in data:
        if dtype == 'entity':
            e = 'atom({}("{}")).'.format(format_for_asp(d[2], 'entity'),
                                         '_'.join(d[3].split()))
            result.append(e)
        else:
            r = 'atom({}("{}","{}")).'.format(format_for_asp(d[4], 'relation'),
                                         '_'.join(d[5].split()), '_'.join(d[6].split()))
            result.append(r)
    return result