# ----------------------------------------------------------------------------
# Copyright (c) 2017--, goenrich development team.
#
# Distributed under the terms of the MIT licence.
# ----------------------------------------------------------------------------

# CODE OBTAINED FROM: https://github.com/jdrudolph/goenrich/
# COPIED HERE BECAUSE GOENRICH IS NOT AVAILABLE THROUGH CONDA

import itertools
import networkx as nx
import pandas as pd

def _tokenize(f):
    token = []
    for line in f:
        if line == '\n':
            yield token
            token = []
        else:
            token.append(line)

def _filter_terms(tokens):
    for token in tokens:
        if token[0] == '[Term]\n':
            yield token[1:]

def _parse_terms(terms):
    for term in terms:
        obsolete = False
        node = {}
        parents = []
        for line in term:
            if line.startswith('id:'):
                id = line[4:-1]
            elif line.startswith('name:'):
                node['name'] = line[6:-1]
            elif line.startswith('namespace:'):
                node['namespace'] = line[11:-1]
            elif line.startswith('is_a:'):
                parents.append(line[6:16])
            elif line.startswith('relationship: part_of'):
                parents.append(line[22:32])
            elif line.startswith('is_obsolete'):
                obsolete = True
                break
        if not obsolete:
            edges = [(p, id) for p in parents] # will reverse edges later
            yield (id, node), edges
        else:
            continue

_filename = 'db/go-basic.obo'

def ontology(file):
    """ read ontology from file
    :param file: file path of file handle
    """
    O = nx.DiGraph()

    if isinstance(file, str):
        f = open(file)
        we_opened_file = True
    else:
        f = file
        we_opened_file = False

    try:
        tokens = _tokenize(f)
        terms = _filter_terms(tokens)
        entries = _parse_terms(terms)
        nodes, edges = zip(*entries)
        O.add_nodes_from(nodes)
        O.add_edges_from(itertools.chain.from_iterable(edges))
        O.graph['roots'] = {data['name'] : n for n, data in O.nodes.items()
                if data['name'] == data['namespace']}
    finally:
        if we_opened_file:
            f.close()

    for root in O.graph['roots'].values():
        for n, depth in nx.shortest_path_length(O, root).items():
            node = O.nodes[n]
            node['depth'] = min(depth, node.get('depth', float('inf')))
    return O.reverse()


"""
parsers for different go-annotation formats
"""
GENE_ASSOCIATION_COLUMNS = ('db', 'db_object_id', 'db_object_symbol',
                            'qualifier', 'go_id', 'db_reference',
                            'evidence_code', 'with_from', 'aspect',
                            'db_object_name', 'db_object_synonym',
                            'db_object_type', 'taxon', 'date', 'assigned_by',
                            'annotation_extension', 'gene_product_form_id')
EXPERIMENTAL_EVIDENCE = ('EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP')


def goa(filename, experimental=True, **kwds):
    """ read go-annotation file

    :param filename: protein or gene identifier column
    :param experimental: use only experimentally validated annotations
    """
    defaults = {'comment': '!',
                'names': GENE_ASSOCIATION_COLUMNS}

    if experimental and 'usecols' in kwds:
        kwds['usecols'] += ('evidence_code',)

    defaults.update(kwds)
    result = pd.read_csv(filename, sep='\t', **defaults)

    if experimental:
        retain_mask = result.evidence_code.isin(EXPERIMENTAL_EVIDENCE)
        result.drop(result.index[~retain_mask], inplace=True)

    return result


def sgd(filename, experimental=False, **kwds):
    """ read yeast genome database go-annotation file
    :param filename: protein or gene identifier column
    :param experimental: use only experimentally validated annotations
    """
    return goa(filename, experimental, **kwds)


GENE2GO_COLUMNS = ('tax_id', 'GeneID', 'GO_ID', 'Evidence', 'Qualifier', 'GO_term', 'PubMed', 'Category')


def gene2go(filename, experimental=False, tax_id=9606, **kwds):
    """ read go-annotation file

    :param filename: protein or gene identifier column
    :param experimental: use only experimentally validated annotations
    :param tax_id: filter according to taxon
    """
    defaults = {'comment': '#',
                'names': GENE2GO_COLUMNS}
    defaults.update(kwds)
    result = pd.read_csv(filename, sep='\t', **defaults)

    retain_mask = result.tax_id == tax_id
    result.drop(result.index[~retain_mask], inplace=True)

    if experimental:
        retain_mask = result.Evidence.isin(EXPERIMENTAL_EVIDENCE)
        result.drop(result.index[~retain_mask], inplace=True)

    return result